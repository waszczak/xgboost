/*!
 * Copyright 2015 by Contributors
 * \file uplift.cc
 * \brief Definition of single-value regression and classification objectives for uplift models.
 * \author Tianqi Chen, Kailong Chen, Peter Foley (Analytics Media Group)
 */
#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../../src/common/math.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(uplift_obj);

// common regressions
// linear regression
struct UpliftLinearSquareLoss {
  static float PredTransform(float x) { return x; }
  static bool CheckLabel(float x) { return true; }
  static float FirstOrderGradient(float predt, float label) { return predt - label; }
  static float SecondOrderGradient(float predt, float label) { return 1.0f; }
  static float ProbToMargin(float base_score) { return base_score; }
  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "rmse"; }
};
// logistic loss for probability regression task
struct UpliftLogisticRegression {
  static float PredTransform(float x) { return common::Sigmoid(x); }
  static bool CheckLabel(float x) { return x >= 0.0f && x <= 1.0f; }
  static float FirstOrderGradient(float predt, float label) { return predt - label; }
  static float SecondOrderGradient(float predt, float label) {
    const float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
  static float ProbToMargin(float base_score) {
    CHECK(base_score > 0.0f && base_score < 1.0f)
    << "base_score must be in (0,1) for logistic loss";
    return -std::log(1.0f / base_score - 1.0f);
  }
  static const char* LabelErrorMsg() {
    return "label must be in [0,1] for logistic regression";
  }
  static const char* DefaultEvalMetric() { return "rmse"; }
};
// logistic loss for binary classification task.
struct UpliftLogisticClassification : public UpliftLogisticRegression {
  static const char* DefaultEvalMetric() { return "error"; }
};
// logistic loss, but predict un-transformed margin
struct UpliftLogisticRaw : public UpliftLogisticRegression {
  static float PredTransform(float x) { return x; }
  static float FirstOrderGradient(float predt, float label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  static float SecondOrderGradient(float predt, float label) {
    const float eps = 1e-16f;
    predt = common::Sigmoid(predt);
    return std::max(predt * (1.0f - predt), eps);
  }
  static const char* DefaultEvalMetric() { return "auc"; }
};

struct UpliftRegLossParam : public dmlc::Parameter<UpliftRegLossParam> {
  float scale_pos_weight;
  int num_output_group;
  // declare parameters
  DMLC_DECLARE_PARAMETER(UpliftRegLossParam) {
    DMLC_DECLARE_FIELD(scale_pos_weight).set_default(1.0f).set_lower_bound(0.0f)
                                        .describe("Scale the weight of positive examples by this factor");
    DMLC_DECLARE_FIELD(num_output_group).set_default(2).set_lower_bound(2)
                                        .describe("Number of output columns (# treatment columns plus one).");
  }
};

// regression los function
template<typename Loss>
class UpliftLossObj : public ObjFunction {
public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float> &betas,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
                     CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
                     CHECK_EQ(betas.size(),info.labels.size())
                       << "UpliftLossObj: label size and pred size does not match";
                     
                     const int nobs = info.num_row;
                     const int ntreat = param_.num_output_group - 1;
                     const int stride = ntreat + 1;
                     
                     out_gpair->resize(betas.size());
                     // check if label in range
                     bool label_correct = true;
                     // start calculating gradient
                     const omp_ulong ndata = static_cast<omp_ulong>(nobs);
#pragma omp parallel for schedule(static)
                     for (omp_ulong i = 0; i < ndata; ++i) {
                       // calculate prediction
                       // first column of betas is constant term (note row major)
                       int idx = stride*i;
                       float y_hat = betas[stride*i];
                       // 1-index j since the first column of labels/betas is skipped
                       for (omp_ulong j = 1; j < stride; ++j) {
                         // y= [1 labels[,-1]].betas
                         // labels/betas is row major
                         idx++;
                         y_hat += info.labels[idx]*betas[idx];
                       }
                       // calculate transformed y_hat
                       float y_trans = Loss::PredTransform(y_hat);
                       float w = info.GetWeight(i);
                       if (info.labels[stride*i] == 1.0f) w *= param_.scale_pos_weight;
                       if (!Loss::CheckLabel(info.labels[stride*i])) label_correct = false;
                       
                       float g_y = Loss::FirstOrderGradient(y_trans, info.labels[stride*i]);
                       float h_y = Loss::SecondOrderGradient(y_trans, info.labels[stride*i]);
                       
                       // first column is constant term
                       out_gpair->at(stride*i) = bst_gpair(g_y * w, h_y * w);
                       for (omp_ulong j = 1; j <= ntreat; ++j) {
                         // chain rule to scale grad/hessian by treatment term
                         float t = info.labels[stride*i+j];
                         out_gpair->at(stride*i+j) = bst_gpair(g_y * w * t, h_y * w * t*t);
                       }
                     }
                     if (!label_correct) {
                       LOG(FATAL) << Loss::LabelErrorMsg();
                     }
                   } 
  const char* DefaultEvalMetric() const override {
    return Loss::DefaultEvalMetric();
  }
  void PredTransform(std::vector<float> *io_preds) override {
    std::vector<float> &preds = *io_preds;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
#pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      preds[j] = Loss::PredTransform(preds[j]);
    }
  }
  float ProbToMargin(float base_score) const override {
    return Loss::ProbToMargin(base_score);
  }
  
protected:
  UpliftRegLossParam param_;
};

// register the ojective functions
DMLC_REGISTER_PARAMETER(UpliftRegLossParam);

XGBOOST_REGISTER_OBJECTIVE(UpliftLinearRegression, "uplift:reg:linear")
  .describe("Linear regression.")
  .set_body([]() { return new UpliftLossObj<UpliftLinearSquareLoss>(); });

XGBOOST_REGISTER_OBJECTIVE(UpliftUpliftLogisticRegression, "uplift:reg:logistic")
  .describe("Logistic regression for probability regression task.")
  .set_body([]() { return new UpliftLossObj<UpliftLogisticRegression>(); });

XGBOOST_REGISTER_OBJECTIVE(UpliftUpliftLogisticClassification, "uplift:binary:logistic")
  .describe("Logistic regression for binary classification task.")
  .set_body([]() { return new UpliftLossObj<UpliftLogisticClassification>(); });

XGBOOST_REGISTER_OBJECTIVE(UpliftUpliftLogisticRaw, "uplift:binary:logitraw")
  .describe("Logistic regression for classification, output score before logistic transformation")
  .set_body([]() { return new UpliftLossObj<UpliftLogisticRaw>(); });


}  // namespace obj
}  // namespace xgboost
