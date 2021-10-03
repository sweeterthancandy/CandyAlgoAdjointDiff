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
            return { "t", "T", "r", "S", "K", "vol" };
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

    using three_addreess_generater_ty = ThreeAddressFunctionGenerator<kernel_ty>;
    three_addreess_generater_ty three_addreess_generater;
    std::shared_ptr<Function> three_addr_func = three_addreess_generater.GenerateInstructionBlock();
    three_addr_func->SetFunctionName("BlackScholesCallOptionTestBareMetalNoDiff");

    using aad_generater_ty = AADFunctionGenerator<kernel_ty>;
    aad_generater_ty aad_generator;
    std::shared_ptr<Function> aad_func = aad_generator.GenerateInstructionBlock();
    aad_func->SetFunctionName("BlackScholesCallOptionTestBareMetal");

    CodeWriter writer;

    std::ofstream out("BlackGenerated.h");

    writer.Emit(out, aad_func);
    writer.Emit(out, three_addr_func);
}

#include "BlackGenerated.h"

#include <chrono>
#include <string>

struct cpu_timer {
    cpu_timer()
        : start_{ std::chrono::high_resolution_clock::now() }
    {}
    std::string format()const {
        return std::to_string(this->count());
    }
    long long count()const
    {
        return (std::chrono::high_resolution_clock::now() - start_).count();
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};


void test_bs() {
#if 1
    double t = 0.0;
    double T = 2.0;
    double r = 0.00;
    double S = 200;
    double K = 200;
    double vol = 0.2;

    double d_t = 0.0;
    double d_T = 0.0;
    double d_r = 0.0;
    double d_S = 0.0;
    double d_K = 0.0;
    double d_vol = 0.0;

    double e = 1e-10;

    std::vector<double> args{ t,T,r,S,K,vol };

    auto k = BlackScholesCallOptionTest::Build<double>{};

    auto from_proto = k.EvaluateVec(args);


    auto from_fun = BlackScholesCallOptionTestBareMetal(
        t, T, r, S, K, vol,
        &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);

    std::cout << "from_proto = " << from_proto << "\n";
    std::cout << "from_fun = " << from_fun << "\n";

    std::cout << "d_t = " << d_t << "\n";
    std::cout << "d_T = " << d_T << "\n";
    std::cout << "d_r = " << d_r << "\n";
    std::cout << "d_S = " << d_S << "\n";
    std::cout << "d_K = " << d_K << "\n";
    std::cout << "d_vol = " << d_vol << "\n";

    for (size_t idx = 0; idx != args.size(); ++idx)
    {
        auto up = args;
        up[idx] += e;
        auto down = args;
        down[idx] -= e;

        auto d = (k.EvaluateVec(up) - k.EvaluateVec(down)) / 2 / e;


        auto d2 = (k.Invoke(BlackScholesCallOptionTestBareMetalNoDiff, up) - k.Invoke(BlackScholesCallOptionTestBareMetalNoDiff, down)) / 2 / e;
        std::cout << "idx => " << d << ", " << d2 << "\n";
    }

    for (volatile size_t N = 1000; N < 1000000; N *= 2)
    {
        std::cout << "N = " << N << "\n";
        cpu_timer analytic_timer;
        for (volatile size_t idx = 0; idx != N; ++idx)
            auto from_proto = k.EvaluateVec(args);
        //BlackScholesCallOptionTestBareMetalNOADD(t, T, r, S, K, vol);
        std::cout << "    analytic took " << analytic_timer.format() << "\n";

        auto analytic_count = analytic_timer.count();

        cpu_timer aad_timer;
        for (volatile size_t idx = 0; idx != N; ++idx)
            auto from_fun = BlackScholesCallOptionTestBareMetal(
                t, T, r, S, K, vol,
                &d_t, &d_T, &d_r, &d_S, &d_K, &d_vol);
        auto aad_count = aad_timer.count();
        std::cout << "    AAD took " << aad_timer.format() << "\n";
        std::cout << "    ratio = " << double(aad_count) / analytic_count << "\n";

    }
#endif
}

int main()
{
    enum { RunDriver = 1 };
    if (RunDriver)
    {
        driver();
    }
    else
    {
        test_bs();
        // run_it();
    }



}