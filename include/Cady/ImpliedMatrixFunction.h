#ifndef INCLUDE_IMPLIED_MATRIX_FUNC_H
#define INCLUDE_IMPLIED_MATRIX_FUNC_H

#include "Cady.h"

namespace Cady {

    struct ImpliedMatrixFunction
    {
        ImpliedMatrixFunction(
            size_t id,
            std::string const& output,
            std::vector<std::shared_ptr<Symbol> > const& args,
            std::vector<std::shared_ptr<Operator> > const& diffs)
            : id_{ id }
            , output_{ output }
            , args_{ args }
            , diffs_{ diffs }
        {}
        static std::shared_ptr< ImpliedMatrixFunction> Make(size_t id, std::shared_ptr<Instruction> const& instr)
        {
            if (auto decl_instr = std::dynamic_pointer_cast<InstructionDeclareVariable>(instr))
            {
                auto expr = decl_instr->as_operator_();
                auto deps = expr->DepthFirstAnySymbolicDependencyOrThisNoRecurse();
                std::vector<std::shared_ptr<Symbol> > input_sym(
                    deps.Set.begin(), deps.Set.end());
                std::vector<std::shared_ptr<Operator> > diff_vec;
                for (auto const& sym : input_sym) {
                    Transform::FoldZero fold_zero;

                    auto diff = fold_zero.Fold(expr->Diff(sym->Name()));
                    diff_vec.push_back(diff);
                }
                return std::make_shared< ImpliedMatrixFunction>(id, decl_instr->LValueName(), input_sym, diff_vec);
            }
            throw std::runtime_error("not implemented");

        }
        std::shared_ptr<InstructionComment> MakeComment()const
        {
            std::vector<std::string> comment_vec;

            comment_vec.push_back("Matrix function F_" + std::to_string(id_) + " => " + output_);
            for (size_t idx = 0; idx != args_.size(); ++idx)
            {
                std::stringstream ss;
                ss << "    " << args_[idx]->Name() << " => ";
                diffs_[idx]->EmitCode(ss);
                std::string comment = ss.str();
                if (comment.back() == '\n')
                {
                    comment.pop_back();
                }
                comment_vec.push_back(comment);


            }

            return std::make_shared<InstructionComment>(comment_vec);
        }

        std::shared_ptr<InstructionDeclareMatrix> MakeMatrix(std::unordered_map<std::string, size_t> const& alloc, bool is_terminal)const
        {
            auto find_slot = [&](std::string const& name)
            {
                auto iter = alloc.find(name);
                if (iter == alloc.end())
                {
                    throw std::runtime_error("bad alloctedd slot");
                }
                return iter->second;
            };

            auto zero = Constant::Make(0.0);
            auto one = Constant::Make(1.0);

            size_t n = alloc.size();

            std::vector<std::shared_ptr<Operator> > diff_col(n, zero);
            for (size_t idx = 0; idx != args_.size(); ++idx)
            {
                auto const& sym = args_[idx];
                auto const& diff = diffs_[idx];

                auto j = find_slot(sym->Name());

                diff_col[j] = diff;
            }



            std::vector<std::vector<std::shared_ptr<Operator> > > matrix(n);
            if (is_terminal)
            {
                for (size_t idx = 0; idx != n; ++idx)
                {
                    matrix[idx].push_back(diff_col[idx]);
                }
            }
            else
            {
                for (auto& row : matrix)
                {
                    row.resize(n + 1, zero);
                }
                // write identity matrix

                for (size_t i = 0; i != n; ++i)
                {
                    matrix[i][i] = one;
                }

                for (size_t idx = 0; idx != n; ++idx)
                {
                    matrix[idx][n] = diff_col[idx];
                }
            }


            std::string name = "adj_matrix_" + std::to_string(id_);
            return std::make_shared<InstructionDeclareMatrix>(name, std::make_shared< SymbolicMatrix>(matrix));

        }


    private:
        size_t id_;
        std::string output_;
        std::vector<std::shared_ptr<Symbol> > args_;
        std::vector<std::shared_ptr<Operator> > diffs_;
    };

} // end namespace Cady

#endif // INCLUDE_IMPLIED_MATRIX_FUNC_H