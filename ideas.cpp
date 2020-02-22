#include <iostream>


/*
        Target, price a cap with the below code

        CAP            = N \sum_{i in {\alpha+1,...,\beta) P(0,T_i) \ro_i Black(K,F(0,T_{i-1},T_i), v_i, 1)
        Black(K,F,v,w) = F w Norm(w d_1(K,F,v)) - K w Norm(w d_2(K,F,v))
        d_1(K,F,v)     = (ln(F/K) + v^2/2 )/v
        d_2(K,F,v)     = (ln(F/K) - v^2/2 )/v
        v_i            = \sigma_{\alpha,\beta} \sqrt{ T_{i-1} }

 */
namespace cady{

namespace Text{
};

namespace CodeGen{


        struct Expression{
                virtual ~Expression()=default;
        };

        struct BinaryOperator : Expression{
        };

        struct Statement{
                virtual ~Statement()=default;
        };

        struct StatementList : Statement{
                void AddStatement(std::shared_ptr<Statement> const& stmt){
                        stmts_.push_back(stmt);
                }
        private:
                std::vector<std::shared_ptr<Statement> > stmts_;
        };

        
        
        struct RValue{
                virtual ~RValue(){}
        };

        struct Assignment : Statement, RValue{
        private:
                std::shared_ptr<NamedVariable> 
        };



} // end namespace



namespace Expression{

        enum OperatorKind{
                OP_ADD,
                OP_SUB,
                OP_MUL,
                OP_DIV,
        };
                
        struct Filtration{};

        class Operator : std::enable_shared_from_this<Operator>{
                virtual double Eval(Filtration const& filt)const=0;
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const=0;
                virtual CodeGenerator::RValue EmitCode(CodeGenerator& CG)const=0;
                // idea
                //virtual void EmitLatex(LatexGenerator& LG)=0;
        };




        enum IntervalEndPointKind{
                Endpoint_Open,
                Endpoint_Close,
        };
        enum IntervalEndPointValueKind{
                Endpoint_Finite,
                Endpoint_NegativeInf,
                Endpoint_PositiveInf,
        };
        enum IntervalOperator{
                IOP_LessThan,
                IOP_LessThanEq,
                IPO_Greaterthan,
                IPO_GreaterEq,
        };


        struct BorelSet{
                virtual ~BorelSet()=default;
                virtual bool Contains(double x)const=0;

                // (a,b)
                static std::shared_ptr<BorelSet> MakeOpen(double a, double b);
                // [a,b]
                static std::shared_ptr<BorelSet> MakeClosed(double a, double b);
                // (a,b]
                static std::shared_ptr<BorelSet> MakeLeftOpen(double a, double b);
                // [a,b)
                static std::shared_ptr<BorelSet> MakeRightOpen(double a, double b);
        };

        // [X,\inf)
        struct ClosedRightPartition : BorelSet{
                ClosedRightPartition(double point)
                        :point_(point)
                {}
                bool Contains(double x)const{
                        return x <= point_;
                }
        private:
                double point_;
        };

        // [X,\inf)
        struct OpenRightPartition : BorelSet{
                OpenRightPartition(double point)
                        :point_(point)
                {}
                bool Contains(double x)const{
                        return x < point_;
                }
        private:
                double point_;
        };

        // (\inf, x]
        struct ClosedRightPartition : BorelSet{
                ClosedRightPartition(double point)
                        :point_(point)
                {}
                bool Contains(double x)const{
                        return point_ <= x;
                }
        private:
                double point_;
        };

        struct Union : BorelSet{
                template<class... Args>
                Union(Args&&... args):vec_{args...}{}
                bool Contains(double x)const{
                        for(auto const& ptr : vec_){
                                if( ptr->Contains(x) ){
                                        return true;
                                }
                        }
                        return false;
                }
        private:
                std::vector<std::shared_ptr<BorelSet> > vec_;
        };

        struct Intersection : BorelSet{
                template<class... Args>
                Intersection(Args&&... args):vec_{args...}{}
                bool Contains(double x)const{
                        for(auto const& ptr : vec_){
                                if( ! ptr->Contains(x) ){
                                        return false;
                                }
                        }
                        return true;
                }
        private:
                std::vector<std::shared_ptr<BorelSet> > vec_;
        };






        
        struct Constant : Operator{
                Constant(double value)
                        :value_{value}
                {}
                virtual double Eval(Filtration const& filt)const override{
                        return value_;
                }
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const{
                        return std::make_shared<Constant>(0.0);
                }
                virtual CodeGenerator::RValue EmitCode(CodeGenerator& CG)const{
                        return CG.InvalidRef();
                }
        private:
                std::string name_;
        };

        struct Symbol : Operator{
                Symbol(std::string const& name)
                        :name_(name)
                {}
                virtual double Eval(Filtration const& filt)const override{
                        return ST.Value(name_);
                }
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const{
                        if( name == name_ ){
                                return std::make_shared<Constant>(1.0);
                        } else {
                                return std::make_shared<Constant>(0.0);
                        }
                }
                virtual CodeGenerator::RValue EmitCode(CodeGenerator& CG)const{
                        return CG.InvalidRef();
                }
        private:
                std::string name_;
        };
        
        
        // need to take care the range is independent of any differnetial
        class IndicatorFunction : Operator{
                virtual double Eval(Filtration const& filt)const override{
                        double value = arg_->Eval(filt);
                        if( domain_->contains(value) ){
                                return 1.0;
                        }
                        return 0.0;
                }
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const{
                        return std::make_shared<Constant>(0.0);
                }
                virtual CodeGenerator::RValue EmitCode(CodeGenerator& CG)const{
                        // here we need to emit function of the form
                        //        double tmp = ...;
                        //        if( a <= tmp && tmp <= b ){
                        //        }
                        return CG.InvalidRef();
                }
        private:
                std::shared_ptr<BorelSet> domain_;
                std::shared_ptr<Operator> arg_;
        };

        struct BinaryOperator : Operator{
                BinaryOperator(OperatorKind op,
                               std::shared_ptr<Operator> left,
                               std::shared_ptr<Operator> right)
                        :op_{op},
                        left_(left),
                        right_(right)
                {}
                virtual double Eval(Filtration const& filt)const override{
                        switch(op_){
                                case OP_ADD:
                                        return left_->Eval(ST) + right_->Eval(ST);
                                case OP_SUB:
                                        return left_->Eval(ST) - right_->Eval(ST);
                                case OP_MUL:
                                        return left_->Eval(ST) * right_->Eval(ST);
                                case OP_DIV:
                                {
                                        auto nume = left_->Eval(ST);
                                        if( nume == 0.0 )
                                                return 0.0;
                                        auto deno = right_->Eval(ST);
                                        if( deno == 0.0 )
                                                throw std::domain_error("divide by zero");
                                        return nume / deno;
                                }
                                default:
                                        throw std::domain_error("unknown op");
                        }
                }
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const{
                        return std::make_shared<BinaryOperator>(OP_ADD,
                                std::make_shared<BinaryOperator>(OP_MUL,
                                        left_,
                                        right_->Diff(name)
                                ),
                                std::make_shared<BinaryOperator>(OP_MUL,
                                        left_->Diff(name),
                                        right_
                                )
                        );
                }
                virtual void EmitCode(CodeGenerator& CG)const{
                        auto left_ref = CG.Generate(left_);
                        auto right_ref = CG.Generate(right_);
                        auto ref = CG.MakeBinaryOp(op_, left_ref, right_ref);
                        return ret;
                }
        private:
                OperatorKind op_;
                std::shared_ptr<Operator> left_;
                std::shared_ptr<Operator> right_;
        };


        struct LinkerHandle : Operator{
                LinkerHandle(std::shared_ptr<Operator> const& expr, std::vector<std::string> const& diff_seq = {})
                        : expr_{expr}
                        , diff_seq_{diff_seq}
                {}
                virtual double Eval(Filtration const& filt)const override{
                        if( double* value = filt.TryFindInformation(shared_from_this())){
                                return *value;
                        } else {
                                auto tmp = expr_;
                                for(auto const& x : diff_seq_ ){
                                        tmp = tmp->Diff(x);
                                }
                                auto value = tmp->Eval(ST);
                                filt.AddInformation(shared_from_this(), value);
                                return value;
                        }
                }
                virtual std::shared_ptr<Operator> Diff(std::string const& name)const{
                        return std::make_shared<LinkerHandle>(expr_, d_depth_ + 1 );
                }
                virtual CodeGenerator::RValue EmitCode(CodeGenerator& CG)const{
                        return CG.InvalidRef();
                }
        private:
                std::shared_ptr<Operator> expr_;
                std::vector<std::string> diff_seq_;
        };

} // end namespace Expression

namespace Frontend{

        using Expr = std::shared_ptr<Expression::Operator>;

        struct Param{
                explicit Param(std::string const& name)
                {
                        expr_ = std::make_shared<Expression::Symbol>(name);
                }
                Expr to_expr()const{ return expr_; }
        private:
                Expr expr_;
        };

        struct Constant{
                explicit Constant(double value)
                {
                        expr_ = std::make_shared<Expression::Constant>(value);
                }
                Expr to_expr()const{ return expr_; }
        private:
                Expr expr_;
        };

        template<class Left, class Right>
        Expr operator+(Left const& left, Right const& right){
                return std::make_shared<Expression::BinaryOperator>(
                        OP_ADD,
                        left.to_expr(),
                        right.to_expr()
                );
        }
        template<class Left, class Right>
        Expr operator*(Left const& left, Right const& right){
                return std::make_shared<Expression::BinaryOperator>(
                        OP_MUL,
                        left.to_expr(),
                        right.to_expr()
                );
        }

        struct Stmt{
                template<class Expr>
                Stmt(Expr&& expr){
                        handle_ = std::make_shared<Expression::LinkerHandle>(expr.to_expr());
                }
                Expr to_expr()const{ return handle_; }
        private:
                Expr handle_;
        };
        
} // end namespace Frontend


// this is all the code needed to turn the expression tree into a c code, or otherwise
struct TransformationPass{
};
struct TransformationManager{
};

} // end namespace cady

namespace Application{

        struct CurveBuilder{
                std::vector<std::shared_ptr<Expression::Operator> > Build(std::vector<std::tuple<double,double> > const& points)const{
                        assert( points.size() >= 2);

                        std::vector<std::shared_ptr<Expression::Operator> > subs;

                        using namespace Expression;

                        size_t idx=0;
                        for(;idx+2<points.size();++idx){
                                subs.push_back(std::make_shared<IndicatorFunction>(
                                        BorelSet::MakeRightOpen( std::get<0>(points[idx]), std::get<0>(points[idx+1])),
                                        BinaryOperator::MakeAdd(
                                                Constant::Make( std::get<1>(points[idx])),
                                                BinaryOperator::MakeMul(
                                                        Symbol::Make("x"),
                                                        Constant::Make( std::get<1>(points[idx+1]) -  std::get<1>(points[idx]))
                                                )
                                        )
                                );

                        }

                }
                
        };


} // end namespace Application

void example_0(){


        // y = a x + b

        using namespace Frontend;

        /*
         */
        Stmt a = Constant(2.0);
        Stmt b = Constant(3.0);
        Stmt x = Param("x");



        auto y = a * x + b;

        auto y_expr = y.Expression();

        auto dy_expr = y_expr->Diff("x");

        


}

int main(){}
