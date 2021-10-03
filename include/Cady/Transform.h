#ifndef INCLUDE_CADY_TRANSFORM_H
#define INCLUDE_CADY_TRANSFORM_H

#include "Cady.h"
#include <map>

namespace Cady{
namespace Transform{

struct FoldZero{

        std::shared_ptr<Operator> Fold(std::shared_ptr<Operator> root){
                if( root->Kind() == OPKind_BinaryOperator ){
                        auto bin_op = static_cast<BinaryOperator*>(root.get());

                        auto left_folded = this->Fold(bin_op->At(0));
                        auto right_folded = this->Fold(bin_op->At(1));

                        auto left_desc  = ConstantDescription{left_folded};
                        auto right_desc = ConstantDescription{right_folded};


                        switch(bin_op->OpKind())
                        {
                                case OP_ADD:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                double sum = 0.0;
                                                sum += left_desc.ValueOrThrow();
                                                sum += right_desc.ValueOrThrow();
                                                return Constant::Make(sum);
                                        }
                                        if( left_desc.IsZero() )
                                                return right_folded;
                                        if( right_desc.IsZero() )
                                                return left_folded;


                                        break;
                                }
                                case OP_SUB:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                double sum = 0.0;
                                                sum += left_desc.ValueOrThrow();
                                                sum -= right_desc.ValueOrThrow();
                                                return Constant::Make(sum);
                                        }

                                        if( right_desc.IsZero())
                                                return left_folded;

                                        if( left_desc.IsZero() ){
                                                return UnaryOperator::UnaryMinus(right_folded);
                                        }
                                        break;
                                }
                                case OP_MUL:
                                {
                                        if( left_desc.IsZero() || right_desc.IsZero() ){
                                                return Constant::Make(0.0);
                                        }
                                        if( left_desc.IsOne() )
                                                return right_folded;
                                        if( right_desc.IsOne() )
                                                return left_folded;
                                        break;
                                }
                                case OP_DIV:
                                {
                                        if( ! left_desc.IsZero() ){
                                                if( right_desc.IsZero() ){
                                                        throw std::domain_error("have divide by zero");
                                                }
                                        } else {
                                                return Constant::Make(0.0);
                                        }
                                        break;
                                }
                                case OP_POW:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                return Constant::Make(
                                                        std::pow(
                                                                left_desc.ValueOrThrow()
                                                                ,right_desc.ValueOrThrow()
                                                        )
                                                );
                                        }

                                        if( left_desc.IsZero() )
                                                return Constant::Make(0.0);

                                        if( right_desc.IsOne() ){
                                                return left_folded;
                                        }
                                        if( right_desc.IsZero() ){
                                                return Constant::Make(1.0);
                                        }
                                        break;
                                }
                        }
                        return std::make_shared<BinaryOperator>(
                                bin_op->OpKind(),
                                left_folded,
                                right_folded);
                }
                
                if( root->Kind() == OPKind_UnaryOperator ){
                        auto unary_op = static_cast<UnaryOperator*>(root.get());

                        auto folded_arg = this->Fold(unary_op->At(0));

                        auto arg_desc  = ConstantDescription{folded_arg};

                        if( arg_desc.IsZero() ){
                                return Constant::Make(0.0);
                        }

                        return UnaryOperator::UnaryMinus(folded_arg);
                }

                if( root->IsNonTerminal() ){
                        for(size_t idx=0;idx!=root->Arity();++idx){
                                auto folded = this->Fold(root->At(idx));
                                root->Rebind(idx, folded);
                        }
                }
                return root;
        }
};


/*
This function makes the expression tree three address code, ie

        *
      /    \
    2      +
          /  \
        3      4


goes to

    tmp0 = 3 + 4
    tmp1 = 2 * tmp0
*/
struct RemapUnique : OperatorTransform {
    explicit RemapUnique(std::string const& prefix = "__symbol_")
        : prefix_{ prefix }
    {
    }
    ~RemapUnique() {
    }
    void mutate_prefix(std::string const& prefix) {
        prefix_ = prefix;
    }
    virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {

        auto candidate = ptr->Clone(shared_from_this());

        auto key = std::make_tuple(
            candidate->NameInvariantOfChildren(),
            candidate->Children()
        );

        auto iter = ops_.find(key);
        if (iter != ops_.end())
            return iter->second;

        if (
            candidate->Kind() != OPKind_EndgenousSymbol &&
#if 0
            candidate->Kind() != OPKind_ExogenousSymbol &&
#endif
            candidate->Kind() != OPKind_Constant)
        {

            std::stringstream ss;
            ss << prefix_ << (ops_.size() + 1);
            auto endogous_sym = EndgenousSymbol::Make(ss.str(), candidate);

            ops_[key] = endogous_sym;
            return endogous_sym;
        }
        else {
            ops_[key] = candidate;
            return candidate;
        }
    }
private:
    std::string prefix_;
    std::map<
        std::tuple<
        std::string,
        std::vector<std::shared_ptr<Operator> >
        >,
        std::shared_ptr<Operator>
    > ops_;
};

} // end namespace Transform
} // end namespace Cady

#endif // INCLUDE_CADY_TRANSFORM_H
