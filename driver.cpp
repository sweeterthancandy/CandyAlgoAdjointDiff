#include "Cady/Cady.h"
#include "Cady/Frontend.h"
#include "Cady/CodeGen.h"

#include <map>
#include <iomanip>

using namespace Cady;
using namespace Cady::CodeGen;

void example_0(){


        Function f("f");
        f.AddArgument("x");
        f.AddArgument("y");

        //auto expr_0 = BinaryOperator::Mul(Log::Make(BinaryOperator::Mul(ExogenousSymbol::Make("x"),ExogenousSymbol::Make("x"))),  Exp::Make(ExogenousSymbol::Make("y")));
        //auto expr_0 = BinaryOperator::Pow(ExogenousSymbol::Make("x"), Constant::Make(2));
        auto expr_0 = Phi::Make(BinaryOperator::Pow(ExogenousSymbol::Make("x"), Constant::Make(3)));

        auto stmt_0 = std::make_shared<EndgenousSymbol>("stmt0", expr_0);

        f.AddStatement(stmt_0);

        std::ofstream fstr("prog.cxx");
        fstr << R"(
#include <cstdio>
#include <cmath>
)";

        StringCodeGenerator cg;
        cg.Emit(fstr, f);
        fstr << R"(

int main(){
        double x_min = 0.1;
        double x_max = +2.0;
        double y_min = -2.0;
        double y_max = +2.0;

        double epsilon = 1e-10;
        double increment = 0.05;

        

        for(double x =x_min; x <= x_max + increment /2; x += increment ){
                for(double y =y_min; y <= y_max + increment /2; y += increment ){
                        double d_x = 0.0;
                        double d_y = 0.0;

                        double value = f(x, &d_x, y, &d_y);

                        double dummy;
                        double x_lower = f(x - epsilon /2 , &dummy, y, &dummy);
                        double x_upper = f(x + epsilon /2 , &dummy, y, &dummy);
                        double x_finite_diff = ( x_upper - x_lower ) / epsilon;
                        double x_residue = d_x - x_finite_diff;
                        
                        double y_lower = f(x, &dummy, y - epsilon /2 , &dummy);
                        double y_upper = f(x, &dummy, y + epsilon /2 , &dummy);
                        double y_finite_diff = ( y_upper - y_lower ) / epsilon;
                        double y_residue = d_y - y_finite_diff;
                        
                        //printf("%f,%f,%f,%f,%f,%f\n", x, y, d_x, d_y, x_finite_diff, x_residue);
                        printf("%f,%f,%f => %f,%f,%f => %f,%f,%f\n", x, y,value, d_x, x_finite_diff,x_residue, d_y, y_finite_diff,y_residue);
                }


        }

}
)";
}

void example_1(){


        Function f("f");
        f.AddArgument("a");
        f.AddArgument("b");
        f.AddArgument("x");


        auto expr_0 = BinaryOperator::Mul( ExogenousSymbol::Make("a"), BinaryOperator::Mul(ExogenousSymbol::Make("x"),  ExogenousSymbol::Make("x")));


        auto stmt_0 = std::make_shared<EndgenousSymbol>("stmt0", expr_0);

        auto expr_1 = BinaryOperator::Add( ExogenousSymbol::Make(stmt_0->Name()), ExogenousSymbol::Make("b"));

        auto stmt_1 = std::make_shared<EndgenousSymbol>("stmt1", expr_1);
        f.AddStatement(stmt_0);
        f.AddStatement(stmt_1);

        std::ofstream fstr("prog.c");
        StringCodeGenerator cg;
        cg.Emit(fstr, f);
        fstr << R"(
#include <stdio.h>
int main(){
        double a = 2.0;
        double b = 3.0;

        double epsilon = 1e-10;
        double increment = 0.05;

        

        for(double x =0.0; x <= 2.0 + increment /2; x += increment ){
                double d_a = 0.0;
                double d_b = 0.0;
                double d_x = 0.0;

                double y = f(a, &d_a, b, &d_b, x, &d_x);

                double dummy;
                double lower = f(a, &dummy, b, &dummy, x - epsilon/2, &dummy);
                double upper = f(a, &dummy, b, &dummy, x + epsilon/2, &dummy);
                double finite_diff = ( upper - lower ) / epsilon;
                double residue = d_x - finite_diff;
                
                printf("%f,%f,%f,%f,%f,%f,%f\n", x, y, d_a, d_b, d_x, finite_diff, residue);


        }

}
)";
}

void black_scholes(){


        Function f("black");
        f.AddArgument("t");
        f.AddArgument("T");
        f.AddArgument("r");
        f.AddArgument("S");
        f.AddArgument("K");
        f.AddArgument("vol");

        auto time_to_expiry = BinaryOperator::Sub(
                ExogenousSymbol::Make("T"),
                ExogenousSymbol::Make("t")
        );

        auto deno = BinaryOperator::Div( 
                Constant::Make(1.0),
                BinaryOperator::Mul(
                        ExogenousSymbol::Make("vol"),
                        BinaryOperator::Pow(
                                time_to_expiry,
                                Constant::Make(0.5)
                        )
                )
        );

        auto d1 = BinaryOperator::Mul(
                deno,
                BinaryOperator::Add(
                        Log::Make(
                                BinaryOperator::Div(
                                        ExogenousSymbol::Make("S"),
                                        ExogenousSymbol::Make("K")
                                )
                        ),
                        BinaryOperator::Mul(
                                BinaryOperator::Add(
                                        ExogenousSymbol::Make("r"),
                                        BinaryOperator::Div(
                                                BinaryOperator::Pow(
                                                        ExogenousSymbol::Make("vol"),
                                                        Constant::Make(2.0)
                                                ),
                                                Constant::Make(2.0)
                                        )
                                ),
                                time_to_expiry
                        )
                )
        );

        auto stmt_0 = std::make_shared<EndgenousSymbol>("stmt0", d1);

        
        auto d2 = BinaryOperator::Sub(
                ExogenousSymbol::Make(stmt_0->Name()),
                BinaryOperator::Mul(
                        ExogenousSymbol::Make("vol"),
                        time_to_expiry
                )
        );


        auto stmt_1 = std::make_shared<EndgenousSymbol>("stmt1", d2);
        
        auto pv = BinaryOperator::Mul(
                ExogenousSymbol::Make("K"),
                Exp::Make(
                        BinaryOperator::Mul(
                                BinaryOperator::Sub(
                                        Constant::Make(0.0),
                                        ExogenousSymbol::Make("r")
                                ),
                                time_to_expiry
                        )
                )
        );
        
        auto stmt_2 = std::make_shared<EndgenousSymbol>("stmt2", pv);

        auto black = BinaryOperator::Sub(
                BinaryOperator::Mul(
                        Phi::Make(stmt_0),
                        ExogenousSymbol::Make("S")
                ),
                BinaryOperator::Mul(
                        Phi::Make(stmt_1),
                        stmt_2
                )
        );

        auto stmt_3 = std::make_shared<EndgenousSymbol>("stmt3", black);


        f.AddStatement(stmt_0);
        f.AddStatement(stmt_1);
        f.AddStatement(stmt_2);
        f.AddStatement(stmt_3);

        std::ofstream fstr("prog.cxx");
        fstr << R"(
#include <cstdio>
#include <cmath>
)";

        StringCodeGenerator cg;
        cg.Emit(fstr, f);
        fstr << R"(

double black_fd(double epsilon, double t, double d_t, double T, double d_T, double r, double d_r, double S, double d_S, double K, double d_K, double vol, double d_vol){
        double dummy;
        double lower = black( t - d_t*epsilon/2 , &dummy, T - d_T*epsilon/2  , &dummy, r - d_r*epsilon/2  , &dummy, S - d_S*epsilon/2  , &dummy, K - d_K*epsilon/2  , &dummy, vol - d_vol*epsilon/2, &dummy);
        double upper = black( t + d_t*epsilon/2 , &dummy, T + d_T*epsilon/2  , &dummy, r + d_r*epsilon/2  , &dummy, S + d_S*epsilon/2  , &dummy, K + d_K*epsilon/2  , &dummy, vol + d_vol*epsilon/2, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        return finite_diff;
}
int main(){
        double t   = 0.0;
        double T   = 10.0;
        double r   = 0.04;
        double S   = 50;
        double K   = 60;
        double vol = 0.2;

        double epsilon = 1e-10;

        double d_t = 0.0;
        double d_T = 0.0;
        double d_r = 0.0;
        double d_S = 0.0;
        double d_K = 0.0;
        double d_vol = 0.0;
        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);

        double d1 = 1/ ( vol * std::sqrt(T - t)) *  ( std::log(S/K) + ( r + vol*vol/2)*(T-t));

        double dummy;
        double lower = black( t - epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double upper = black( t + epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        double residue = d_t - finite_diff;

        printf("%f,%f,%f,%f,%f,%f => %f,%f => %f,%f,%f\n", t, T, r, S, K, vol, value, d1, d_t, finite_diff, residue);

        printf("d[t]  ,%f,%f\n", d_t  ,  black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[T]  ,%f,%f\n", d_T  ,  black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[r]  ,%f,%f\n", d_r  ,  black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0));
        printf("d[S]  ,%f,%f\n", d_S  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0));
        printf("d[K]  ,%f,%f\n", d_K  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0));
        printf("d[vol],%f,%f\n", d_vol,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1));
        

}
)";
}
















void black_scholes_frontend(){

        using namespace Frontend;
        using Frontend::Log;
        using Frontend::Exp;
        using Frontend::Phi;


        Function f("black");
        f.AddArgument("t");
        f.AddArgument("T");
        f.AddArgument("r");
        f.AddArgument("S");
        f.AddArgument("K");
        f.AddArgument("vol");

        auto d1    = f.AddStatement(Stmt("d1"   , (1.0 / ( Var("vol") * ((Var("T") - Var("t")) ^ 0.5) )) * ( Log(Var("S") / "K") +   ("r" + ( Var("vol") ^ 2.0 ) / 2 ) * (Var("T") - Var("t")) )));
        auto d2    = f.AddStatement(Stmt("d2"   , d1 - "vol" * (Var("T") - Var("t"))));
        auto pv    = f.AddStatement(Stmt("pv"   , "K" * Exp( -Var("r") * ( Var("T") - Var("t") ) )));
        auto black = f.AddStatement(Stmt("black", Phi(d1) * "S" - Phi(d2) * pv));

        std::ofstream fstr("prog.cxx");
        fstr << R"(
#include <cstdio>
#include <cmath>
)";

        StringCodeGenerator cg;
        cg.Emit(fstr, f);
        fstr << R"(

double black_fd(double epsilon, double t, double d_t, double T, double d_T, double r, double d_r, double S, double d_S, double K, double d_K, double vol, double d_vol){
        double dummy;
        double lower = black( t - d_t*epsilon/2 , &dummy, T - d_T*epsilon/2  , &dummy, r - d_r*epsilon/2  , &dummy, S - d_S*epsilon/2  , &dummy, K - d_K*epsilon/2  , &dummy, vol - d_vol*epsilon/2, &dummy);
        double upper = black( t + d_t*epsilon/2 , &dummy, T + d_T*epsilon/2  , &dummy, r + d_r*epsilon/2  , &dummy, S + d_S*epsilon/2  , &dummy, K + d_K*epsilon/2  , &dummy, vol + d_vol*epsilon/2, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        return finite_diff;
}
int main(){
        double t   = 0.0;
        double T   = 10.0;
        double r   = 0.04;
        double S   = 50;
        double K   = 60;
        double vol = 0.2;

        double epsilon = 1e-10;

        double d_t = 0.0;
        double d_T = 0.0;
        double d_r = 0.0;
        double d_S = 0.0;
        double d_K = 0.0;
        double d_vol = 0.0;
        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);

        double d1 = 1/ ( vol * std::sqrt(T - t)) *  ( std::log(S/K) + ( r + vol*vol/2)*(T-t));

        double dummy;
        double lower = black( t - epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double upper = black( t + epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        double residue = d_t - finite_diff;

        printf("%f,%f,%f,%f,%f,%f => %f,%f => %f,%f,%f\n", t, T, r, S, K, vol, value, d1, d_t, finite_diff, residue);

        printf("d[t]  ,%f,%f\n", d_t  ,  black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[T]  ,%f,%f\n", d_T  ,  black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[r]  ,%f,%f\n", d_r  ,  black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0));
        printf("d[S]  ,%f,%f\n", d_S  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0));
        printf("d[K]  ,%f,%f\n", d_K  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0));
        printf("d[vol],%f,%f\n", d_vol,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1));
        

}
)";
}

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

        using Frontend::Phi;
        using Frontend::Exp;
        using Frontend::Pow;
        using Frontend::Log;

} // end namespace MathFunctions

struct BlackScholesCallOption{
        template<class Double>
        struct Build{
                Double Evaluate(
                        Double t,
                        Double T,
                        Double r,
                        Double S,
                        Double K,
                        Double vol )const
                {
                        using MathFunctions::Phi;
                        using MathFunctions::Exp;
                        using MathFunctions::Pow;
                        using MathFunctions::Log;

                        Double d1 = ((1.0 / ( vol * Pow((T - t),0.5) )) * ( Log(S / K) +   (r + ( Pow(vol,2.0) ) / 2 ) * (T - t) ));
                        Double d2 = d1 - vol * (T - t);
                        Double pv = K * Exp( -r * ( T - t ) );
                        Double black = Phi(d1) * S - Phi(d2) * pv;
                        return black;
                }
        };
};


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

void black_scholes_template(){
        auto black_eval = BlackScholesCallOption::Build<double>{};

        double t   = 0.0;
        double T   = 10.0;
        double r   = 0.04;
        double S   = 50;
        double K   = 60;
        double vol = 0.2;

        std::cout << "black_eval(t,T,r,S,K,vol) => " << black_eval.Evaluate(t,T,r,S,K,vol) << "\n"; // __CandyPrint__(cxx-print-scalar,black_eval(t,T,r,S,K,vol))


        
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate( 
                DoubleKernel::BuildFromExo("t"),
                DoubleKernel::BuildFromExo("T"),
                DoubleKernel::BuildFromExo("r"),
                DoubleKernel::BuildFromExo("S"),
                DoubleKernel::BuildFromExo("K"),
                DoubleKernel::BuildFromExo("vol")
        );


        Function f("black");
        f.AddArgument("t");
        f.AddArgument("T");
        f.AddArgument("r");
        f.AddArgument("S");
        f.AddArgument("K");
        f.AddArgument("vol");

        using namespace Frontend;

        std::unordered_set< std::shared_ptr<Operator > > seen;
        struct StackFrame{
                explicit StackFrame(std::shared_ptr<EndgenousSymbol > op)
                        : Op{op}
                {
                        auto deps_set = Op->EndgenousDependencies();
                        Deps.assign(deps_set.begin(), deps_set.end());
                }
                std::shared_ptr<EndgenousSymbol > Op;
                std::vector<std::shared_ptr<EndgenousSymbol > > Deps;
        };
        std::vector<StackFrame> stack{StackFrame{std::reinterpret_pointer_cast<EndgenousSymbol>(as_black.as_operator_())}};
        for(size_t ttl=1000;stack.size() && ttl;--ttl){
                auto& frame = stack.back();
                if( frame.Deps.size() == 0 ){
                        if( seen.count(frame.Op) == 0 ){
                                seen.insert(frame.Op);
                                auto black = f.AddStatement(frame.Op);
                                #if 0
                                std::cout << "----------TERMINAL--------------\n";
                                frame.Op->Display();
                                #endif
                        }
                        stack.pop_back();
                        continue;
                }
                auto dep = frame.Deps.back();
                frame.Deps.pop_back();

                stack.push_back(StackFrame{dep});

        }


        std::ofstream fstr("prog.cxx");
        fstr << R"(
#include <cstdio>
#include <cmath>
#include <iostream>
#include <boost/timer/timer.hpp>
)";

        StringCodeGenerator cg;
        cg.Emit(fstr, f);
        fstr << R"(

double black_fd(double epsilon, double t, double d_t, double T, double d_T, double r, double d_r, double S, double d_S, double K, double d_K, double vol, double d_vol){
        double dummy;
        double lower = black( t - d_t*epsilon/2 , &dummy, T - d_T*epsilon/2  , &dummy, r - d_r*epsilon/2  , &dummy, S - d_S*epsilon/2  , &dummy, K - d_K*epsilon/2  , &dummy, vol - d_vol*epsilon/2, &dummy);
        double upper = black( t + d_t*epsilon/2 , &dummy, T + d_T*epsilon/2  , &dummy, r + d_r*epsilon/2  , &dummy, S + d_S*epsilon/2  , &dummy, K + d_K*epsilon/2  , &dummy, vol + d_vol*epsilon/2, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        return finite_diff;
}
int main(){
        double t   = 0.0;
        double T   = 10.0;
        double r   = 0.04;
        double S   = 50;
        double K   = 60;
        double vol = 0.2;

        double epsilon = 1e-10;

        double d_t = 0.0;
        double d_T = 0.0;
        double d_r = 0.0;
        double d_S = 0.0;
        double d_K = 0.0;
        double d_vol = 0.0;
        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);

        double d1 = 1/ ( vol * std::sqrt(T - t)) *  ( std::log(S/K) + ( r + vol*vol/2)*(T-t));

        double dummy;
        double lower = black( t - epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double upper = black( t + epsilon/2 , &dummy, T  , &dummy, r  , &dummy, S  , &dummy, K  , &dummy, vol, &dummy);
        double finite_diff = ( upper - lower ) / epsilon;
        double residue = d_t - finite_diff;

        printf("%f,%f,%f,%f,%f,%f => %f,%f => %f,%f,%f\n", t, T, r, S, K, vol, value, d1, d_t, finite_diff, residue);

        printf("d[t]  ,%f,%f\n", d_t  ,  black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[T]  ,%f,%f\n", d_T  ,  black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0));
        printf("d[r]  ,%f,%f\n", d_r  ,  black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0));
        printf("d[S]  ,%f,%f\n", d_S  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0));
        printf("d[K]  ,%f,%f\n", d_K  ,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0));
        printf("d[vol],%f,%f\n", d_vol,  black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1));
        
        // time profile
        for(volatile size_t N = 100;;N*=2){
                boost::timer::cpu_timer timer;
                for(volatile size_t idx=0;idx!=N;++idx){
                        double value = black( t  , &d_t, T  , &d_T, r  , &d_r, S  , &d_S, K  , &d_K, vol, &d_vol);
                }
                std::string ad_time = timer.format(4, "%w");
                timer.start();
                for(volatile size_t idx=0;idx!=N;++idx){
                        black_fd(epsilon, t, 1, T  , 0, r  , 0, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 1, r  , 0, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 1, S  , 0, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 1, K  , 0, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 1, vol, 0);
                        black_fd(epsilon, t, 0, T  , 0, r  , 0, S  , 0, K  , 0, vol, 1);
                }
                std::string fd_time = timer.format(4, "%w");
                std::cout << N << "," << fd_time << "," << ad_time << "\n";
        }

}
)";
}


struct RemoveEndgenousFolder{
        std::shared_ptr<Operator> Fold(std::shared_ptr<Operator> root){
                if( root->Kind() == OPKind_EndgenousSymbol ){
                        auto as_endgenous = std::reinterpret_pointer_cast<EndgenousSymbol>(root);
                        return this->Fold(as_endgenous->Expr());
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

struct RemapUnique : OperatorTransform{
        virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr){

                auto candidate = ptr->Clone(shared_from_this());

                auto key = std::make_tuple(
                        candidate->NameInvariantOfChildren(),
                        candidate->Children()
                        );

                auto iter = ops_.find(key);
                if( iter != ops_.end() )
                        return iter->second;

                if( candidate->Kind() != OPKind_EndgenousSymbol &&
                    candidate->Kind() != OPKind_ExogenousSymbol &&
                    candidate->Kind() != OPKind_Constant )
                {

                        std::stringstream ss;
                        ss << "__symbol_" << ops_.size();
                        auto endogous_sym = EndgenousSymbol::Make(ss.str(), candidate); 
                        
                        ops_[key] = endogous_sym;
                        return endogous_sym;
                } else {
                        ops_[key] = candidate;
                        return candidate;
                }
        }
private:
        std::map<
                std::tuple<
                        std::string,
                        std::vector<std::shared_ptr<Operator> > 
                >,
                std::shared_ptr<Operator>
        > ops_;
};




void black_scholes_template_opt(){

        
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate( 
                DoubleKernel::BuildFromExo("t"),
                DoubleKernel::BuildFromExo("T"),
                DoubleKernel::BuildFromExo("r"),
                DoubleKernel::BuildFromExo("S"),
                DoubleKernel::BuildFromExo("K"),
                DoubleKernel::BuildFromExo("vol")
        );

        SymbolTable ST;
        ST("t"  , 0.0);
        ST("T"  , 10.0);
        ST("r"  , 0.04);
        ST("S"  , 50);
        ST("K"  , 60);
        ST("vol", 0.2);

        RemoveEndgenousFolder remove_endogous;
        Transform::FoldZero constant_fold;

        auto black_expr = as_black.as_operator_();

        std::cout << "--------- black_expr -----------\n";
        //black_expr->Display();
        std::cout << "black_expr->Eval(ST) => " << black_expr->Eval(ST) << "\n"; // __CandyPrint__(cxx-print-scalar,black_expr->Eval(ST))

        auto removed_endo = remove_endogous.Fold(black_expr);

        auto params = std::vector<std::string>{ "t", "T", "r", "S", "K", "vol" };

        std::vector<std::shared_ptr<Operator> > ticker;
        for(auto const& s : params){
                auto raw_diff = removed_endo->Diff(s);
                ticker.push_back(raw_diff);
        }

        auto unique_mapper = std::make_shared<RemapUnique>();

        std::ostream& out = std::cout;

        out << "double black(";
        for(size_t idx=0;idx!=params.size();++idx){
                if( idx != 0 ){
                        out << ", ";
                }
                out << "double " << params[idx] << ", double* d_" << params[idx];
        }
        out << "){\n";

        std::unordered_set<std::shared_ptr<Operator> > seen;
        std::shared_ptr<EndgenousSymbol> return_;
        for(size_t idx=0;idx!=ticker.size();++idx){
                auto constant_folded = constant_fold.Fold(ticker[idx]);
                
                auto unique          = constant_folded->Clone(unique_mapper);

                auto dependents = unique->DepthFirstAnySymbolicDependency();

                for(auto const& dep : dependents.DepthFirst){
                        // first emit all the expressions we need
                        if( seen.count(dep) > 0 )
                                continue;
                        seen.insert(dep);

                        out << "    double " << std::left << std::setw(15) << dep->Name() << " = ";
                        dep->Expr()->EmitCode(out);
                        out << ";\n";
                }

                out << "    *d_" << params[idx] << " = " << dependents.DepthFirst.back()->Name() << ";\n";



                //unique->Display();
        }

        auto constant_folded = constant_fold.Fold(removed_endo);
        
        auto unique          = constant_folded->Clone(unique_mapper);

        auto dependents = unique->DepthFirstAnySymbolicDependency();

        for(auto const& dep : dependents.DepthFirst){
                // first emit all the expressions we need
                if( seen.count(dep) > 0 )
                        continue;
                seen.insert(dep);

                out << "    double " << std::left << std::setw(15) << dep->Name() << " = ";
                dep->Expr()->EmitCode(out);
                out << ";\n";
        }
        out << "    return " << dependents.DepthFirst.back()->Name() << ";\n";
        out << "}\n";






}



int main(){
        //black_scholes();
        //black_scholes_frontend();
        //black_scholes_template();
        black_scholes_template_opt();

}
