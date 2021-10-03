#ifndef INCLUDE_CADY_FRONTEND_H
#define INCLUDE_CADY_FRONTEND_H

#include "Cady.h"


namespace Cady{
namespace Frontend{

        struct ImbueWith{};

        struct WithOperators{
                WithOperators(std::shared_ptr<Operator> impl)
                        :impl_(impl)
                {}
                std::shared_ptr<Operator> as_operator_()const{
                        return impl_;
                }
        private:
                std::shared_ptr<Operator> impl_;
        };

        namespace Detail{

                template<size_t Precedence>
                struct PrecedenceDevice : PrecedenceDevice<Precedence-1>{};
                template<>
                struct PrecedenceDevice<0>{};

                template<
                        class T,
                        class = std::void_t<
                                typename std::enable_if<
                                        std::is_same<const char*, typename std::decay<T>::type>::value ||
                                        std::is_same<char*      , typename std::decay<T>::type>::value ||
                                        std::is_same<std::string, typename std::decay<T>::type>::value
                                >::type
                        >
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<7>&&){
                        return ExogenousSymbol::Make(t);
                }
                template<
                        class T,
                        class = std::void_t<
                                typename std::enable_if<
                                        std::is_floating_point<typename std::decay<T>::type>::value ||
                                        std::is_integral<typename std::decay<T>::type>::value
                                >::type
                        >
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<8>&&){
                        return Constant::Make(t);
                }
                template<
                        class T,
                        class = std::void_t<decltype(std::declval<T>()->as_operator_())>
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<9>&&){
                        return t->as_operator_();
                }
                template<
                        class T,
                        class = std::void_t<decltype(std::declval<T>().as_operator_())>
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<10>&&){
                        return t.as_operator_();
                }
                template<
                        class T,
                        class = std::void_t<
                                typename std::enable_if<
                                        std::is_same<
                                                std::shared_ptr<Operator>,
                                                typename std::decay<T>::type
                                        >::value
                                >::type
                        >
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<20>&&){
                        return t;
                }

        }  // end namespace Detail
        
        template<class T>
        std::shared_ptr<Operator> AsOperator(T&& t){
                return Detail::AsOperatorImpl(
                        std::forward<T>(t),
                        Detail::PrecedenceDevice<100>{}
                );
        }

        #define FRONTEND_DEFINE_OPERATOR(LEXICAL_TOKEN, MAPPED_FUNCTION) \
        template<                                                        \
                class L,                                                 \
                class R,                                                 \
                class = std::void_t<                                   \
                        decltype( AsOperator(std::declval<L>()) ),       \
                        decltype( AsOperator(std::declval<R>()) )        \
                >                                                        \
        >                                                                \
        auto operator LEXICAL_TOKEN(L&& l, R&& r){                       \
                return WithOperators{                                    \
                        BinaryOperator::MAPPED_FUNCTION(                 \
                                 AsOperator(l),                          \
                                 AsOperator(r)                           \
                        )                                                \
                };                                                       \
        }
        FRONTEND_DEFINE_OPERATOR(+, Add)
        FRONTEND_DEFINE_OPERATOR(-, Sub)
        FRONTEND_DEFINE_OPERATOR(*, Mul)
        FRONTEND_DEFINE_OPERATOR(/, Div)
        FRONTEND_DEFINE_OPERATOR(^, Pow)

        template<
                class Arg,
                class = std::void_t<
                        decltype( AsOperator(std::declval<Arg>()) )
                >
        >
        auto operator -(Arg&& arg){
                return WithOperators{
                        UnaryOperator::UnaryMinus(
                                 AsOperator(arg)
                        )
                };
        }

        inline auto Var(std::string const& name){
                return WithOperators{ExogenousSymbol::Make(name)};
        }
        template<class Expr>
        inline auto Break(std::string const& name, Expr&& expr){
                return WithOperators{EndgenousSymbol::Make(name, AsOperator(expr))};
        }
        template<class T>
        inline auto Log(T&& arg){
                return WithOperators{ Log::Make( AsOperator(arg) ) };
        }
        template<class T>
        inline auto Sin(T&& arg){
                return WithOperators{ Sin::Make( AsOperator(arg) ) };
        }
        template<class T>
        inline auto Cos(T&& arg){
                return WithOperators{ Cos::Make( AsOperator(arg) ) };
        }
        template<class T>
        inline auto Exp(T&& arg){
                return WithOperators{ Exp::Make( AsOperator(arg) ) };
        }
        template<class T>
        inline auto Phi(T&& arg){
                return WithOperators{ Phi::Make( AsOperator(arg) ) };
        }


        template<class L, class R>
        inline auto Pow(L&& l, R&& r){
                return WithOperators{ 
                        BinaryOperator::Pow( AsOperator(l), AsOperator(r) )
                };
        }

        template<class L, class R>
        inline auto Min(L&& l, R&& r) {
            return WithOperators{
                    BinaryOperator::Min(AsOperator(l), AsOperator(r))
            };
        }
        template<class L, class R>
        inline auto Max(L&& l, R&& r) {
            return WithOperators{
                    BinaryOperator::Max(AsOperator(l), AsOperator(r))
            };
        }

        template<
            class Cond,
            class IfTrue,
            class IfFalse>
        inline auto If(
            Cond&& cond,
            IfTrue&& if_true,
            IfFalse&& if_false)
        {
            return WithOperators{
                    If::Make(
                        AsOperator(cond),
                        AsOperator(if_true()),
                        AsOperator(if_false()))
            };
        }

        template<class Arg>
        inline auto Stmt(std::string const& name, Arg&& arg){
                auto ptr = std::make_shared<EndgenousSymbol>(name, AsOperator(arg));
                return ptr;
        }

        struct Double : WithOperators{
                template<class Expr>
                // non-explicit so we can assign more matural
                //      Double d = ...;
                Double(Expr&& expr)
                        : WithOperators{AsOperator(std::forward<Expr>(expr))}
                {}
        };
        #if 0
        struct Var{
                Var(std::string const& name):
                        impl_{ExogenousSymbol::Make(name)}
                {}
                std::shared_ptr<Operator> as_operator_()const{
                        return impl_;
                }
        private:
                std::shared_ptr<Operator> impl_;
        };
        #endif


} // end namespace Frontend
} // end namespace Cady

namespace Cady{
/*
        Here we want to make a massive distincition between lvalue and rvalue

        each lvalue is single assignment, and maps to statements in the code,
        the idea is that we might have something like this

                a = f(x,y,z)
                b = a * ( a - 1 ) * (a -2 ) * ( a -3 )

        mathematically, the above is nothing more than the expanded expression,
        however we want to allow the creation of statements. 
 */
struct DoubleKernelImpl{
        virtual ~DoubleKernelImpl()=default;
        virtual std::shared_ptr<Operator> as_operator_()const=0;
};
struct DoubleKernelOperator : DoubleKernelImpl{
        DoubleKernelOperator(std::shared_ptr<Operator> ptr)
                :operator_(ptr)
        {}
        virtual std::shared_ptr<Operator> as_operator_()const{ return operator_; }
private:
        std::shared_ptr<Operator> operator_;
};
struct DoubleKernel : Frontend::ImbueWith{
        static std::string Tag(){
                static size_t counter = 0;
                std::stringstream ss;
                ss << "__statement_" << counter;
                ++counter;
                return ss.str();
        }
        template< class Expr >
        DoubleKernel(Expr&& expr)
                : impl_{std::make_shared<DoubleKernelOperator>(EndgenousSymbol::Make(Tag(), Frontend::AsOperator(expr)))}
        {}
        struct Dispatch_Exo{};
        DoubleKernel( Dispatch_Exo&&, std::shared_ptr<Operator> const& op)
                : impl_{std::make_shared<DoubleKernelOperator>(op)}
        {}
        static DoubleKernel BuildFromExo(std::string const& name){
                return DoubleKernel(Dispatch_Exo{}, std::make_shared<ExogenousSymbol>(name)); 
        }
        std::shared_ptr<Operator> as_operator_()const{
                return impl_->as_operator_();
        }
private:
        std::shared_ptr<DoubleKernelImpl> impl_;
};
namespace MathFunctions{

        inline double Phi(double x){
                return std::erfc(-x/std::sqrt(2))/2;
        }
        inline double Exp(double x){
                return std::exp(x);
        }
        inline double Pow(double x, double y){
                return std::pow(x,y);
        }
        inline double Log(double x){
                return std::log(x);
        }
        inline double Min(double l, double r) {
            return std::min(l, r);
        }
        inline double Max(double l, double r) {
            return std::max(l, r);
        }
        template<class IfTrue, class IfFalse>
        inline double If(
            double cond,
            IfTrue&& if_true,
            IfFalse&& if_false)
        {
            if (!!cond)
            {
                return if_true();
            }
            else
            {
                return if_false();
            }
        }

        using Frontend::Phi;
        using Frontend::Exp;
        using Frontend::Pow;
        using Frontend::Log;
        using Frontend::Min;
        using Frontend::Max;
        using Frontend::If;

} // end namespace MathFunctions
} // end namespace Cady

#endif // INCLUDE_CADY_FRONTEND_H
