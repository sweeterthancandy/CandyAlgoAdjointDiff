#include <iostream>

#include "Cady/CodeGen.h"
#include "Cady/Frontend.h"
#include "Cady/Transform.h"
#include "Cady/Cady.h"
#include "Cady/SymbolicMatrix.h"
#include "Cady/Instruction.h"
#include "Cady/ImpliedMatrixFunction.h"
#include "Cady/AADFunctionGenerator.h"

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

            Double d1 = ((1.0 / (vol * Pow((T - t), 0.5))) * (Log(S / K) + (r + (Pow(vol, 2.0)) / 2) * (T - t)));
            Double d2 = d1 - vol * (T - t);
            Double pv = K * Exp(-r * (T - t));
            Double black = Phi(d1) * S - Phi(d2) * pv;
            return black;
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
            return "BlackScholesCallOptionTest";
        }
    };
};




void driver()
{
    using generater_ty = AADFunctionGenerator<BlackScholesCallOptionTest>;
    generater_ty generator;
    std::string function_def = generator.GenerateString();
    std::cout << function_def << "\n";
}


double BlackScholesCallOptionTestBareMetalNOADD(double t, double T, double r, double S, double K, double vol)
{
    // Matrix function F_0 => __symbol_11
    //     vol => 1.000000

    double const __symbol_11 = vol;

    // Matrix function F_1 => __symbol_2
    //     T => 1.000000

    double const __symbol_2 = T;

    // Matrix function F_2 => __symbol_1
    //     t => 1.000000

    double const __symbol_1 = t;

    // Matrix function F_3 => __symbol_3
    //     __symbol_2 => 1.000000
    //     __symbol_1 => -1.000000

    double const __symbol_3 = ((__symbol_2)-(__symbol_1));

    // Matrix function F_4 => __symbol_23
    //     __symbol_3 => ((0.500000)*(std::pow(__symbol_3, -0.500000)))

    double const __symbol_23 = std::pow(__symbol_3, 0.500000);

    // Matrix function F_5 => __symbol_24
    //     __symbol_11 => __symbol_23
    //     __symbol_23 => __symbol_11

    double const __symbol_24 = ((__symbol_11) * (__symbol_23));

    // Matrix function F_6 => __symbol_26
    //     __symbol_24 => ((-1.000000)/(std::pow(__symbol_24, 2.000000)))

    double const __symbol_26 = ((1.000000) / (__symbol_24));

    // Matrix function F_7 => __symbol_18
    //     S => 1.000000

    double const __symbol_18 = S;

    // Matrix function F_8 => __symbol_8
    //     K => 1.000000

    double const __symbol_8 = K;

    // Matrix function F_9 => __symbol_19
    //     __symbol_18 => ((__symbol_8)/(std::pow(__symbol_8, 2.000000)))
    //     __symbol_8 => (((-(__symbol_18)))/(std::pow(__symbol_8, 2.000000)))

    double const __symbol_19 = ((__symbol_18) / (__symbol_8));

    // Matrix function F_10 => __symbol_20
    //     __symbol_19 => ((1.000000)/(__symbol_19))

    double const __symbol_20 = std::log(__symbol_19);

    // Matrix function F_11 => __symbol_4
    //     r => 1.000000

    double const __symbol_4 = r;

    // Matrix function F_12 => __symbol_14
    //     __symbol_11 => ((2.000000)*(__symbol_11))

    double const __symbol_14 = std::pow(__symbol_11, 2.000000);

    // Matrix function F_13 => __symbol_15
    //     __symbol_14 => ((2.000000)/(4.000000))

    double const __symbol_15 = ((__symbol_14) / (2.000000));

    // Matrix function F_14 => __symbol_16
    //     __symbol_4 => 1.000000
    //     __symbol_15 => 1.000000

    double const __symbol_16 = ((__symbol_4)+(__symbol_15));

    // Matrix function F_15 => __symbol_17
    //     __symbol_16 => __symbol_3
    //     __symbol_3 => __symbol_16

    double const __symbol_17 = ((__symbol_16) * (__symbol_3));

    // Matrix function F_16 => __symbol_21
    //     __symbol_20 => 1.000000
    //     __symbol_17 => 1.000000

    double const __symbol_21 = ((__symbol_20)+(__symbol_17));

    // Matrix function F_17 => __symbol_27
    //     __symbol_26 => __symbol_21
    //     __symbol_21 => __symbol_26

    double const __symbol_27 = ((__symbol_26) * (__symbol_21));

    // Matrix function F_18 => __statement_0
    //     __symbol_27 => 1.000000

    double const __statement_0 = __symbol_27;

    // Matrix function F_19 => __symbol_33
    //     __statement_0 => ((std::exp((-(((0.500000)*(std::pow(__statement_0, 2.000000)))))))/(2.506628))

    double const __symbol_33 = std::erfc(-(__statement_0) / std::sqrt(2)) / 2;

    // Matrix function F_20 => __symbol_34
    //     __symbol_33 => __symbol_18
    //     __symbol_18 => __symbol_33

    double const __symbol_34 = ((__symbol_33) * (__symbol_18));

    // Matrix function F_21 => __symbol_12
    //     __symbol_11 => __symbol_3
    //     __symbol_3 => __symbol_11

    double const __symbol_12 = ((__symbol_11) * (__symbol_3));

    // Matrix function F_22 => __symbol_29
    //     __symbol_12 => -1.000000
    //     __statement_0 => 1.000000

    double const __symbol_29 = ((__statement_0)-(__symbol_12));

    // Matrix function F_23 => __statement_1
    //     __symbol_29 => 1.000000

    double const __statement_1 = __symbol_29;

    // Matrix function F_24 => __symbol_31
    //     __statement_1 => ((std::exp((-(((0.500000)*(std::pow(__statement_1, 2.000000)))))))/(2.506628))

    double const __symbol_31 = std::erfc(-(__statement_1) / std::sqrt(2)) / 2;

    // Matrix function F_25 => __symbol_5
    //     __symbol_4 => (-(1.000000))

    double const __symbol_5 = (-(__symbol_4));

    // Matrix function F_26 => __symbol_6
    //     __symbol_5 => __symbol_3
    //     __symbol_3 => __symbol_5

    double const __symbol_6 = ((__symbol_5) * (__symbol_3));

    // Matrix function F_27 => __symbol_7
    //     __symbol_6 => std::exp(__symbol_6)

    double const __symbol_7 = std::exp(__symbol_6);

    // Matrix function F_28 => __symbol_9
    //     __symbol_8 => __symbol_7
    //     __symbol_7 => __symbol_8

    double const __symbol_9 = ((__symbol_8) * (__symbol_7));

    // Matrix function F_29 => __statement_2
    //     __symbol_9 => 1.000000

    double const __statement_2 = __symbol_9;

    // Matrix function F_30 => __symbol_32
    //     __symbol_31 => __statement_2
    //     __statement_2 => __symbol_31

    double const __symbol_32 = ((__symbol_31) * (__statement_2));

    // Matrix function F_31 => __symbol_35
    //     __symbol_32 => -1.000000
    //     __symbol_34 => 1.000000

    double const __symbol_35 = ((__symbol_34)-(__symbol_32));

    // Matrix function F_32 => __statement_3
    //     __symbol_35 => 1.000000

    double const __statement_3 = __symbol_35;

    return __statement_3;
}



double BlackScholesCallOptionTestBareMetal(double t, double T, double r, double S, double K, double vol, double* d_t = nullptr, double* d_T = nullptr, double* d_r = nullptr, double* d_S = nullptr, double* d_K = nullptr, double* d_vol = nullptr)
{
    // Matrix function F_0 => __symbol_11
    //     vol => 1.000000

    double const __symbol_11 = vol;

    // Matrix function F_1 => __symbol_2
    //     T => 1.000000

    double const __symbol_2 = T;

    // Matrix function F_2 => __symbol_1
    //     t => 1.000000

    double const __symbol_1 = t;

    // Matrix function F_3 => __symbol_3
    //     __symbol_2 => 1.000000
    //     __symbol_1 => -1.000000

    double const __symbol_3 = ((__symbol_2)-(__symbol_1));

    // Matrix function F_4 => __symbol_23
    //     __symbol_3 => ((0.500000)*(std::pow(__symbol_3, -0.500000)))

    double const __symbol_23 = std::pow(__symbol_3, 0.500000);

    // Matrix function F_5 => __symbol_24
    //     __symbol_23 => __symbol_11
    //     __symbol_11 => __symbol_23

    double const __symbol_24 = ((__symbol_11) * (__symbol_23));

    // Matrix function F_6 => __symbol_26
    //     __symbol_24 => ((-1.000000)/(std::pow(__symbol_24, 2.000000)))

    double const __symbol_26 = ((1.000000) / (__symbol_24));

    // Matrix function F_7 => __symbol_18
    //     S => 1.000000

    double const __symbol_18 = S;

    // Matrix function F_8 => __symbol_8
    //     K => 1.000000

    double const __symbol_8 = K;

    // Matrix function F_9 => __symbol_19
    //     __symbol_18 => ((__symbol_8)/(std::pow(__symbol_8, 2.000000)))
    //     __symbol_8 => (((-(__symbol_18)))/(std::pow(__symbol_8, 2.000000)))

    double const __symbol_19 = ((__symbol_18) / (__symbol_8));

    // Matrix function F_10 => __symbol_20
    //     __symbol_19 => ((1.000000)/(__symbol_19))

    double const __symbol_20 = std::log(__symbol_19);

    // Matrix function F_11 => __symbol_4
    //     r => 1.000000

    double const __symbol_4 = r;

    // Matrix function F_12 => __symbol_14
    //     __symbol_11 => ((2.000000)*(__symbol_11))

    double const __symbol_14 = std::pow(__symbol_11, 2.000000);

    // Matrix function F_13 => __symbol_15
    //     __symbol_14 => ((2.000000)/(4.000000))

    double const __symbol_15 = ((__symbol_14) / (2.000000));

    // Matrix function F_14 => __symbol_16
    //     __symbol_4 => 1.000000
    //     __symbol_15 => 1.000000

    double const __symbol_16 = ((__symbol_4)+(__symbol_15));

    // Matrix function F_15 => __symbol_17
    //     __symbol_16 => __symbol_3
    //     __symbol_3 => __symbol_16

    double const __symbol_17 = ((__symbol_16) * (__symbol_3));

    // Matrix function F_16 => __symbol_21
    //     __symbol_20 => 1.000000
    //     __symbol_17 => 1.000000

    double const __symbol_21 = ((__symbol_20)+(__symbol_17));

    // Matrix function F_17 => __symbol_27
    //     __symbol_26 => __symbol_21
    //     __symbol_21 => __symbol_26

    double const __symbol_27 = ((__symbol_26) * (__symbol_21));

    // Matrix function F_18 => __statement_0
    //     __symbol_27 => 1.000000

    double const __statement_0 = __symbol_27;

    // Matrix function F_19 => __symbol_33
    //     __statement_0 => ((std::exp((-(((0.500000)*(std::pow(__statement_0, 2.000000)))))))/(2.506628))

    double const __symbol_33 = std::erfc(-(__statement_0) / std::sqrt(2)) / 2;

    // Matrix function F_20 => __symbol_34
    //     __symbol_33 => __symbol_18
    //     __symbol_18 => __symbol_33

    double const __symbol_34 = ((__symbol_33) * (__symbol_18));

    // Matrix function F_21 => __symbol_12
    //     __symbol_11 => __symbol_3
    //     __symbol_3 => __symbol_11

    double const __symbol_12 = ((__symbol_11) * (__symbol_3));

    // Matrix function F_22 => __symbol_29
    //     __statement_0 => 1.000000
    //     __symbol_12 => -1.000000

    double const __symbol_29 = ((__statement_0)-(__symbol_12));

    // Matrix function F_23 => __statement_1
    //     __symbol_29 => 1.000000

    double const __statement_1 = __symbol_29;

    // Matrix function F_24 => __symbol_31
    //     __statement_1 => ((std::exp((-(((0.500000)*(std::pow(__statement_1, 2.000000)))))))/(2.506628))

    double const __symbol_31 = std::erfc(-(__statement_1) / std::sqrt(2)) / 2;

    // Matrix function F_25 => __symbol_5
    //     __symbol_4 => (-(1.000000))

    double const __symbol_5 = (-(__symbol_4));

    // Matrix function F_26 => __symbol_6
    //     __symbol_5 => __symbol_3
    //     __symbol_3 => __symbol_5

    double const __symbol_6 = ((__symbol_5) * (__symbol_3));

    // Matrix function F_27 => __symbol_7
    //     __symbol_6 => std::exp(__symbol_6)

    double const __symbol_7 = std::exp(__symbol_6);

    // Matrix function F_28 => __symbol_9
    //     __symbol_8 => __symbol_7
    //     __symbol_7 => __symbol_8

    double const __symbol_9 = ((__symbol_8) * (__symbol_7));

    // Matrix function F_29 => __statement_2
    //     __symbol_9 => 1.000000

    double const __statement_2 = __symbol_9;

    // Matrix function F_30 => __symbol_32
    //     __symbol_31 => __statement_2
    //     __statement_2 => __symbol_31

    double const __symbol_32 = ((__symbol_31) * (__statement_2));

    // Matrix function F_31 => __symbol_35
    //     __symbol_34 => 1.000000
    //     __symbol_32 => -1.000000

    double const __symbol_35 = ((__symbol_34)-(__symbol_32));

    // Matrix function F_32 => __statement_3
    //     __symbol_35 => 1.000000

    double const __statement_3 = __symbol_35;

    // //////////////
    // Starting AAD matrix
    // //////////////

    double const __symbol_75 = (-(__symbol_4));

    double const __symbol_40 = ((__symbol_2)-(__symbol_1));

    double const __symbol_77 = ((__symbol_5) * (__symbol_3));

    double const __symbol_79 = std::exp(__symbol_6);

    double const __symbol_59 = std::pow(__symbol_3, 0.500000);

    double const __symbol_61 = ((__symbol_11) * (__symbol_23));

    double const __symbol_63 = ((1.000000) / (__symbol_24));

    double const __symbol_53 = ((__symbol_18) / (__symbol_8));

    double const __symbol_55 = std::log(__symbol_19);

    double const __symbol_43 = std::pow(__symbol_11, 2.000000);

    double const __symbol_45 = ((__symbol_14) / (2.000000));

    double const __symbol_48 = ((__symbol_4)+(__symbol_15));

    double const __symbol_50 = ((__symbol_16) * (__symbol_3));

    double const __symbol_57 = ((__symbol_20)+(__symbol_17));

    double const __symbol_65 = ((__symbol_26) * (__symbol_21));

    double const __symbol_85 = ((__symbol_11) * (__symbol_3));

    double const __symbol_87 = ((__statement_0)-(__symbol_12));

    double const __symbol_110 = std::erfc(-(__statement_1) / std::sqrt(2)) / 2;

    double const __symbol_112 = ((__symbol_31) * (-1.000000));

    double const __symbol_113 = ((__symbol_8) * (__symbol_112));

    double const __symbol_114 = ((__symbol_79) * (__symbol_113));

    double const __symbol_115 = ((__symbol_5) * (__symbol_114));

    double const __symbol_90 = std::pow(__statement_1, 2.000000);

    double const __symbol_91 = ((0.500000) * (__symbol_90));

    double const __symbol_92 = (-(__symbol_91));

    double const __symbol_93 = std::exp(__symbol_92);

    double const __symbol_94 = ((__symbol_93) / (2.506628));

    double const __symbol_81 = ((__symbol_8) * (__symbol_7));

    double const __symbol_84 = ((__statement_2) * (-1.000000));

    double const __symbol_95 = ((__symbol_94) * (__symbol_84));

    double const __symbol_108 = ((-1.000000) * (__symbol_95));

    double const __symbol_109 = ((__symbol_11) * (__symbol_108));

    double const __symbol_116 = ((__symbol_115)+(__symbol_109));

    double const __symbol_68 = std::pow(__statement_0, 2.000000);

    double const __symbol_69 = ((0.500000) * (__symbol_68));

    double const __symbol_70 = (-(__symbol_69));

    double const __symbol_71 = std::exp(__symbol_70);

    double const __symbol_72 = ((__symbol_71) / (2.506628));

    double const __symbol_73 = ((__symbol_72) * (__symbol_18));

    double const __symbol_96 = ((__symbol_95)+(__symbol_73));

    double const __symbol_106 = ((__symbol_26) * (__symbol_96));

    double const __symbol_107 = ((__symbol_16) * (__symbol_106));

    double const __symbol_117 = ((__symbol_116)+(__symbol_107));

    double const __symbol_103 = std::pow(__symbol_3, -0.500000);

    double const __symbol_104 = ((0.500000) * (__symbol_103));

    double const __symbol_98 = std::pow(__symbol_24, 2.000000);

    double const __symbol_99 = ((-1.000000) / (__symbol_98));

    double const __symbol_97 = ((__symbol_21) * (__symbol_96));

    double const __symbol_100 = ((__symbol_99) * (__symbol_97));

    double const __symbol_101 = ((__symbol_11) * (__symbol_100));

    double const __symbol_105 = ((__symbol_104) * (__symbol_101));

    double const __symbol_118 = ((__symbol_117)+(__symbol_105));

    double const result_d_t = ((-1.000000) * (__symbol_118));

    if (d_t) { *d_t = result_d_t; }

    double const result_d_T = ((__symbol_117)+(__symbol_105));

    if (d_T) { *d_T = result_d_T; }

    double const __symbol_121 = (-(1.000000));

    double const __symbol_120 = ((__symbol_3) * (__symbol_114));

    double const __symbol_122 = ((__symbol_121) * (__symbol_120));

    double const __symbol_119 = ((__symbol_3) * (__symbol_106));

    double const result_d_r = ((__symbol_122)+(__symbol_119));

    if (d_r) { *d_r = result_d_r; }

    double const __symbol_128 = std::erfc(-(__statement_0) / std::sqrt(2)) / 2;

    double const __symbol_125 = std::pow(__symbol_8, 2.000000);

    double const __symbol_126 = ((__symbol_8) / (__symbol_125));

    double const __symbol_123 = ((1.000000) / (__symbol_19));

    double const __symbol_124 = ((__symbol_123) * (__symbol_106));

    double const __symbol_127 = ((__symbol_126) * (__symbol_124));

    double const result_d_S = ((__symbol_33)+(__symbol_127));

    if (d_S) { *d_S = result_d_S; }

    double const __symbol_133 = ((__symbol_7) * (__symbol_112));

    double const __symbol_130 = (-(__symbol_18));

    double const __symbol_131 = ((__symbol_130) / (__symbol_125));

    double const __symbol_132 = ((__symbol_131) * (__symbol_124));

    double const result_d_K = ((__symbol_133)+(__symbol_132));

    if (d_K) { *d_K = result_d_K; }

    double const __symbol_140 = ((__symbol_3) * (__symbol_108));

    double const __symbol_138 = ((2.000000) * (__symbol_11));

    double const __symbol_136 = ((2.000000) / (4.000000));

    double const __symbol_137 = ((__symbol_136) * (__symbol_119));

    double const __symbol_139 = ((__symbol_138) * (__symbol_137));

    double const __symbol_141 = ((__symbol_140)+(__symbol_139));

    double const __symbol_134 = ((__symbol_23) * (__symbol_100));

    double const result_d_vol = ((__symbol_141)+(__symbol_134));

    if (d_vol) { *d_vol = result_d_vol; }

    return __statement_3;

}
double BlackScholesCallOptionTestBareMetalNoDiff(double t, double T, double r, double S, double K, double vol)
{
    return BlackScholesCallOptionTestBareMetal(t, T, r, S, K, vol);
}

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

    double t       = 0.0;
    double T       = 10.0;
    double r       = 0.04;
    double S       = 50;
    double K       = 60;
    double vol     = 0.2;

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

       
        auto d2 = (k.Invoke(BlackScholesCallOptionTestBareMetalNoDiff,up) - k.Invoke(BlackScholesCallOptionTestBareMetalNoDiff, down)) / 2 / e;
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
}

int main()
{
    enum{ RunDriver = 0};
    if( RunDriver )
    {
        driver();
    }
    else
    {
        test_bs();
        // run_it();
    }
    
    

}