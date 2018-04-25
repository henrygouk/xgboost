/*!
 * Copyright 2018 by Contributors
 * \file tree_loss_term.h
 * \brief Used for implementing a loss term specific to decision trees
 * \author Henry Gouk
 */
#ifndef XGBOOST_TREE_OMEGA_TERM_H_
#define XGBOOST_TREE_OMEGA_TERM_H_

#include <dmlc/registry.h>
#include <functional>
#include <string>
#include <vector>

namespace xgboost {

namespace tree {
  //TODO(henrygouk): move complete definition of GradStats to this file once design is finalised.
struct GradStats;

class TreeOmegaTerm {
 public:
  static TreeOmegaTerm* Create(const std::string &name);
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& args);
  virtual double CalcSplitGain(tree::GradStats left, tree::GradStats right) const = 0;
  virtual double CalcWeight(tree::GradStats stats) const = 0;
  virtual double CalcGain(tree::GradStats stats) const = 0;
};

struct TreeOmegaTermReg
    : public dmlc::FunctionRegEntryBase<TreeOmegaTermReg,
                                        std::function<TreeOmegaTerm* ()> > {
};

/*!
 * \brief Macro to register tree Omega term.
 *
 * \code
 * // example of registering a Omega term for trees
 * XGBOOST_REGISTER_TREE_OMEGA_TERM(SomeOmega, "someOmega")
 * .describe("Some Omega term")
 * .set_body([]() {
 *     return new SomeOmega();
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_TREE_OEMGA_TERM(UniqueId, Name)                   \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::tree::TreeOmegaTermReg&               \
  __make_ ## TreeOmegaTermReg ## _ ## UniqueId ## __ =                    \
      ::dmlc::Registry< ::xgboost::tree::TreeOmegaTermReg>::Get()->__REGISTER__(Name)

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_OMEGA_H_
