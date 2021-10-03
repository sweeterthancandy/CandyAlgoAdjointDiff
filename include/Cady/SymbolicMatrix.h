#ifndef INCLUDE_SYMBOLIC_MATRIX_H
#define INCLUDE_SYMBOLIC_MATRIX_H

#include "Cady.h"

namespace Cady {

    struct SymbolicMatrix
    {
        SymbolicMatrix(
            std::vector<std::vector<std::shared_ptr<Operator> > > const& matrix)
        {
            cols_ = matrix.front().size();
            rows_ = matrix.size();
            matrix_ = matrix;
        }
        std::shared_ptr<SymbolicMatrix> Multiply(SymbolicMatrix const& that)
        {
            /*
            when we multiply a nxp and a pxq matrix, we get a n x q
            */
            if (cols_ != that.rows_)
            {
                throw std::runtime_error("cna't multiple matricies");
            }

            size_t result_rows = rows_;
            size_t p = cols_;
            size_t result_cols = that.cols_;

            std::vector<std::vector<std::shared_ptr<Operator> > > result;
            for (size_t i = 0; i != result_rows; ++i)
            {
                result.emplace_back();
                for (size_t j = 0; j != result_cols; ++j)
                {
                    std::shared_ptr<Operator> head;
                    for (size_t k = 0; k != p; ++k)
                    {
                        auto const& left = this->At(i, k);
                        auto const& right = that.At(k, j);
                        auto term = BinaryOperator::MulOpt(left, right);
                        if (head)
                        {
                            head = BinaryOperator::AddOpt(head, term);
                        }
                        else
                        {
                            head = term;
                        }


                    }

#if 0
                   Transform::FoldZero fold_zero;
                   result.back().push_back(fold_zero.Fold(head));
#else
                   result.back().push_back(head);
#endif
                }
            }
            return std::make_shared< SymbolicMatrix>(result);

        }
        std::shared_ptr<Operator> const& At(size_t i, size_t j)const
        {
            return matrix_.at(i).at(j);
        }

        std::vector<std::vector<std::shared_ptr<Operator> > > const& get_impl()const { return matrix_; }
    private:
        size_t rows_;
        size_t cols_;
        std::vector<std::vector<std::shared_ptr<Operator> > > matrix_;
    };


} // end namespace Cady

#endif // INCLUDE_SYMBOLIC_MATRIX_H