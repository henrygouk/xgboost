/*!
 * Copyright 2018 by Contributors
 * \file tree_omega_term.cc
 * \brief Registry of tree omega terms.
 */
#include <xgboost/tree_omega_term.h>
#include <dmlc/registry.h>
#include "param.h"

#include "../common/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::tree::TreeOmegaTermReg);
}  // namespace dmlc

namespace xgboost {
namespace tree {

TreeOmegaTerm* TreeOmegaTerm::Create(const std::string& name) {
  auto *e = ::dmlc::Registry< ::xgboost::tree::TreeOmegaTermReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown TreeOmegaTerm " << name;
  }
  return (e->body)();
}

void TreeOmegaTerm::Init(const std::vector<std::pair<std::string, std::string> >& args) {}

static double Sqr(double x) { return x * x; }

class NoOmega : public TreeOmegaTerm {
 public:
  double CalcSplitGain(GradStats left, GradStats right) const override {
    return CalcGain(left) + CalcGain(right);
  }

  double CalcWeight(GradStats stats) const override {
    return -stats.sum_grad / stats.sum_hess;
  }

  double CalcGain(GradStats stats) const override {
    return Sqr(stats.sum_grad) / stats.sum_hess;
  }
};

struct WeightDecayParam : public dmlc::Parameter<WeightDecayParam> {
  double reg_lambda;
  double reg_gamma;

  DMLC_DECLARE_PARAMETER(WeightDecayParam) {
    DMLC_DECLARE_FIELD(reg_lambda)
      .set_lower_bound(0.0)
      .set_default(1.0)
      .describe("L2 regularization on leaf weight");
    DMLC_DECLARE_FIELD(reg_gamma)
      .set_lower_bound(0.0f)
      .set_default(0.0f)
      .describe("Cost incurred by adding a new leaf node to the tree");
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_gamma, gamma);
  }
};

DMLC_REGISTER_PARAMETER(WeightDecayParam);

class WeightDecay : public TreeOmegaTerm {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    m_params.InitAllowUnknown(args);
  }

  double CalcSplitGain(GradStats left, GradStats right) const override {
    return CalcGain(left) + CalcGain(right);
  }

  double CalcWeight(GradStats stats) const override {
    return -stats.sum_grad / (stats.sum_hess + m_params.reg_lambda);
  }

  double CalcGain(GradStats stats) const override {
    return Sqr(stats.sum_grad) / (stats.sum_hess + m_params.reg_lambda) - m_params.reg_gamma;
  }

 private:
  WeightDecayParam m_params;
};

XGBOOST_REGISTER_TREE_OEMGA_TERM(NoOmega, "no_omega")
.describe("Use no Omega term")
.set_body([]() {
    return new NoOmega();
  });

XGBOOST_REGISTER_TREE_OEMGA_TERM(WeightDecay, "weight_decay")
.describe("Use an L2 penalty term for the weights and a cost per leaf node")
.set_body([]() {
    return new WeightDecay();
  });

} // namespace tree
} // namespace xgboost