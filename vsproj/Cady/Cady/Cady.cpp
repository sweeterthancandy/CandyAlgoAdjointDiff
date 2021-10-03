#include <iostream>

#include "Cady/CodeGen.h"
#include "Cady/Frontend.h"
#include "Cady/Transform.h"
#include "Cady/Cady.h"
#include "Cady/SymbolicMatrix.h"
#include "Cady/Instruction.h"
#include "Cady/ImpliedMatrixFunction.h"
#include "Cady/AADFunctionGenerator.h"
#include "Cady/CodeWriter.h"
#include "Cady/CpuTimer.h"

#include <map>
#include <functional>

using namespace Cady;


struct BlackScholesCallOptionTest {
    template<class Double>
    struct Build {
        Double Evaluate(
            Double t,
            Double T,
            Double r,
            Double S,
            Double K,
            Double vol)const
        {
            using MathFunctions::Phi;
            using MathFunctions::Exp;
            using MathFunctions::Pow;
            using MathFunctions::Log;
            using MathFunctions::Max;
            using MathFunctions::Min;


            return Max(S - K, 0.0);
#if 0
            auto on_expiry = ( t == T );
            return If(on_expiry)
                .Then([&]() {
                    Double df = Exp(-r * T);
                    Double F = S * Exp(r * T);
                    Double std = vol * Pow(T, 0.5);
                    Double d = Log(F / K) / std;
                    Double d1 = d + 0.5 * std;
                    Double d2 = d1 - std;
                    Double nd1 = Phi(d1);
                    Double nd2 = Phi(d2);
                    Double c = df * (F * nd1 - K * nd2);
                    return c;
                }).Else([&]() {
                    return Max(S - k, 0.0);
                });
#endif


           

#if 0

            Double x0 = df * F * std;
            Double x1 = d * d1 * d2;
            Double x2 = df * d1 * c;
            Double x3 = c * nd1;
            Double x4 = Log(x0) * F + Log(x3) * Log(x2);
            Double x5 = c * x4 + x0;
            Double x6 = x0 * df + x2;
            Double x7 = x6 * c * nd2;
            Double x8 = Log(x7) * x1 + x2;

            //Double x9 = Log(x0) * Log(x1) * Log(x2) * Log(x3) * Log(x4) + Log(x5) * Log(x6) * Log(x7) * Log(x8);
            Double x9 = Log(x8) * Log(x7) + x5;
            Double x10 = x9 * (x0 + x1 + x2 + x7 * x8);
            Double x11 = Log(x10 + 1);
            Double x12 = x11 / (x11 + 1);
            Double x13 = (x0 * x1 * x2 * x3 * x4) / (x5 * x6 * x7 * x8) + (x9 * x10 * x11 * x11 * x12);

            Double x14 = x13 * x13;
            Double x15 = ( x14 + 1 )* x14;

            return x13;
#endif
        }
        Double EvaluateVec(std::vector<Double> const& args)
        {
            enum { NumArgs = 6 };
            if (args.size() != NumArgs)
            {
                throw std::runtime_error("bad number of args");
            }
            return Evaluate(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5]);
        }
        template<class F>
        Double Invoke(F&& f, std::vector<Double> const& args)
        {
            enum { NumArgs = 6 };
            if (args.size() != NumArgs)
            {
                throw std::runtime_error("bad number of args");
            }
            return f(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5]);
        }
        std::vector<std::string> Arguments()const
        {
            return { "t", "T", "r", "S", "K", "vol"};
        }
        std::string Name()const
        {
            return "BlackScholesCallOptionTestBareMetal";
        }
    };
};




void driver()
{
    // first write no diff version
    using kernel_ty = BlackScholesCallOptionTest;

    

    

    std::vector<
        std::pair<
        std::string,
        std::shared_ptr< FunctionGenerator>
        >
    > ticker;



    ticker.emplace_back("BlackScholesSimple", std::make_shared<SimpleFunctionGenerator<kernel_ty>>());
    ticker.emplace_back("BlackScholesSingleExpr", std::make_shared< SingleExprFunctionGenerator<kernel_ty>>());
    ticker.emplace_back("BlackScholesThreeAddress", std::make_shared< ThreeAddressFunctionGenerator<kernel_ty>>());
    //ticker.emplace_back("BlackScholesThreeAddressFwd", std::make_shared< ForwardDiffFunctionGenerator<kernel_ty>>());

    using aad_generater_ty = AADFunctionGenerator<kernel_ty>;
    ticker.emplace_back("BlackScholesThreeAddressAADFwd", std::make_shared<aad_generater_ty>(aad_generater_ty::AADPT_Forwards));
    ticker.emplace_back("BlackScholesThreeAddressAAD", std::make_shared<aad_generater_ty>(aad_generater_ty::AADPT_Backwards));

    std::vector< std::shared_ptr<Function> > to_emit;

    for (auto const& p : ticker)
    {
        auto const& func_name = p.first;
        auto const& generator = p.second;
        cpu_timer timer;
        auto func = generator->GenerateInstructionBlock();
        std::cout << std::setw(30) << func_name << " took " << timer.format() << "\n";
        func->SetFunctionName(func_name);

        to_emit.push_back(func);
        

    }

    
    CodeWriter writer;

    std::ofstream out("BlackGenerated.h");
    for (auto const& func : to_emit)
    {
        writer.Emit(out, func);;
    }
}
    
   
    

#include "BlackGenerated.h"

#include <chrono>
#include <string>

#if 1


double BlackScholesThreeAddressAADNoDiff(double t, double T, double r, double S, double K, double vol)
{
    double d_t = 0.0;
    double d_T = 0.0;
    double d_r = 0.0;
    double d_S = 0.0;
    double d_K = 0.0;
    double d_vol = 0.0;
    return BlackScholesThreeAddressAAD(t, T, r, S, K, vol, &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);
}

double BlackScholesThreeAddressAADFwdNoDiff(double t, double T, double r, double S, double K, double vol)
{
    double d_t = 0.0;
    double d_T = 0.0;
    double d_r = 0.0;
    double d_S = 0.0;
    double d_K = 0.0;
    double d_vol = 0.0;
    return BlackScholesThreeAddressAADFwd(t, T, r, S, K, vol, &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);
}
#if 0
double BlackScholesThreeAddressFwdNodiff(double t, double T, double r, double S, double K, double vol)
{
    double d_t = 0.0;
    double d_T = 0.0;
    double d_r = 0.0;
    double d_S = 0.0;
    double d_K = 0.0;
    double d_vol = 0.0;
    return BlackScholesThreeAddressFwd(t, T, r, S, K, vol, &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);
}
#endif

double BlackScholesProto(double t, double T, double r, double S, double K, double vol)
{
    return BlackScholesCallOptionTest::Build<double>{}.Evaluate(t, T, r, S, K, vol);
}


void test_bs() {

    using func_ptr_ty = double(*)(double, double, double, double, double, double);
    std::vector<
        std::pair<std::string, func_ptr_ty>
    > bs_world{
        {"BlackScholesProto", BlackScholesProto},
        {"BlackScholesSimple",BlackScholesSimple},
        { "BlackScholesSingleExpr",BlackScholesSingleExpr },
        { "BlackScholesThreeAddress",BlackScholesThreeAddress },
       // { "BlackScholesThreeAddressFwd",BlackScholesThreeAddressFwdNodiff },
        { "BlackScholesThreeAddressAADFwd",BlackScholesThreeAddressAADFwdNoDiff },
        { "BlackScholesThreeAddressAAD",BlackScholesThreeAddressAADNoDiff },
    };

    double t = 0.0;
    double T = 2.0;
    double r = 0.00;
    double S = 201;
    double K = 200;
    double vol = 0.2;

    for (auto const& p : bs_world)
    {
        auto name = p.first;
        auto ptr = p.second;
        std::cout << std::setw(30) << name << " => " << ptr(t, T, r, S, K, vol)  << "\n";
    }


    using diff_func_ptr_ty = double(*)(double, double, double, double, double, double, double*, double*, double*, double*, double*, double*);
    std::vector<
        std::pair<std::string, diff_func_ptr_ty>
    > diff_bs_world = {
          //{ "BlackScholesThreeAddressFwd",BlackScholesThreeAddressFwd},
          { "BlackScholesThreeAddressAADFwd",BlackScholesThreeAddressAADFwd},
        { "BlackScholesThreeAddressAAD",BlackScholesThreeAddressAAD},
    };

    for (auto const& p : diff_bs_world)
    {
        auto name = p.first;
        auto ptr = p.second;

        double d_t = 0.0;
        double d_T = 0.0;
        double d_r = 0.0;
        double d_S = 0.0;
        double d_K = 0.0;
        double d_vol = 0.0;

        double e = 1e-10;

        ptr(
            t, T, r, S, K, vol,
            &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);
        std::vector<double> aad_d_list{ d_t,d_T,d_r,d_S,d_K,d_vol };

        auto k = BlackScholesCallOptionTest::Build<double>{};

        std::vector<double> args{ t,T,r,S,K,vol };
        for (size_t idx = 0; idx != args.size(); ++idx)
        {
            auto up = args;
            up[idx] += e;
            auto down = args;
            down[idx] -= e;

            auto d = (k.EvaluateVec(up) - k.EvaluateVec(down)) / 2 / e;
            auto d_from_func = aad_d_list[idx];

            auto diff = d_from_func - d;
            std::cout << "numeric=" << d << ", " << name << "=" << d_from_func << ", diff=" << diff << "\n";

        }
    }



    


    size_t num_evals = 10000000;
    for (auto const& p : bs_world)
    {
        auto name = p.first;
        auto ptr = p.second;
        cpu_timer timer;
        for (volatile size_t idx = 0; idx != num_evals; ++idx)
            ptr(t, T, r, S, K, vol);
        std::cout << std::setw(30) << name << " => " << timer.format() << "\n";
    }

}
#endif

int main()
{
    driver();
    test_bs();
    
    

}