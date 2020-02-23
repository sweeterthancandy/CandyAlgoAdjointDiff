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
                        class = std::__void_t<
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
                        class = std::__void_t<
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
                        class = std::__void_t<decltype(std::declval<T>()->as_operator_())>
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<9>&&){
                        return t->as_operator_();
                }
                template<
                        class T,
                        class = std::__void_t<decltype(std::declval<T>().as_operator_())>
                >
                std::shared_ptr<Operator> AsOperatorImpl(T&& t, PrecedenceDevice<10>&&){
                        return t.as_operator_();
                }
                template<
                        class T,
                        class = std::__void_t<
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
                class = std::__void_t<                                   \
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
                class = std::__void_t<
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
        template<class T>
        inline auto Log(T&& arg){
                return WithOperators{ Log::Make( AsOperator(arg) ) };
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

#endif // INCLUDE_CADY_FRONTEND_H
