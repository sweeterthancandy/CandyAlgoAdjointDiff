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

#include <array>
#include <map>
#include <functional>

using namespace Cady;

/*
def ko_call_option_analytic(x, K, tau, r, sigma, barrier) :
    """
    From Shreve
    """
    import math
    import scipy.stats
    def N(x) :
    return scipy.stats.norm.cdf(x)

    def factor_plus(tau, s) :
    return 1 / sigma / math.sqrt(tau) * (math.log(s) + (r + sigma * *2 / 2) * tau)

    def factor_minus(tau, s) :
    return 1 / sigma / math.sqrt(tau) * (math.log(s) + (r - sigma * *2 / 2) * tau)

    tmp0 = x * (N(factor_plus(tau, x / K)) - N(factor_plus(tau, x / barrier)))
    tmp1 = -math.exp(-r * tau) * K * (N(factor_minus(tau, x / K)) - N(factor_minus(tau, x / barrier)))
    tmp2 = -barrier * (x / barrier) * *(-2 * r / sigma * *2) * (N(factor_plus(tau, barrier * *2 / K / x)) - N(factor_plus(tau, barrier / x)))
    tmp3_inner = N(factor_minus(tau, barrier * *2 / K / x)) - N(factor_minus(tau, barrier / x))
    tmp3 = math.exp(-r * tau) * K * (x / barrier) * *(-2 * r / sigma * *2 + 1) * tmp3_inner
    return tmp0 + tmp1 + tmp2 + tmp3

*/



namespace KoBarrierOption
{
    struct FactorPlus {

        template<class Double>
        struct Build {
            Double Evaluate(
                Double sigma,
                Double r,
                Double tau,
                Double s)const
            {
                using MathFunctions::Pow;
                using MathFunctions::Log;
                return 1 / sigma / Pow(tau, 0.5) * (Log(s) + (r + Pow(sigma, 2.0) / 2) * tau);
            }
            std::vector<std::string> Arguments()const
            {
                return { "sigma", "r", "tau", "s" };
            }
            std::string Name()const
            {
                return "__FactorPlus";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
        };
    };

    struct FactorMinus{

        template<class Double>
        struct Build {
            Double Evaluate(
                Double sigma,
                Double r,
                Double tau,
                Double s)const
            {
                using MathFunctions::Pow;
                using MathFunctions::Log;
                return 1 / sigma / Pow(tau, 0.5) * (Log(s) + (r - Pow(sigma, 2.0) / 2) * tau);
            }
            std::vector<std::string> Arguments()const
            {
                return { "sigma", "r", "tau", "s" };
            }
            std::string Name()const
            {
                return "__FactorMinus";
            }

            Double EvaluateVec(std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return Evaluate(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
            template<class F>
            Double Invoke(F&& f, std::vector<Double> const& args)
            {
                enum { NumArgs = 4 };
                if (args.size() != NumArgs)
                {
                    throw std::runtime_error("bad number of args");
                }
                return f(
                    args[0],
                    args[1],
                    args[2],
                    args[3]);
            }
        };
    };

    

    struct KoBarrierCallOption {

        template<class Double>
        struct Build {
            Double Evaluate(
                Double x,
                Double K,
                Double tau,
                Double r,
                Double sigma,
                Double B)const
            {
                using MathFunctions::Phi;
                using MathFunctions::Exp;
                using MathFunctions::Call;
                using MathFunctions::Pow;
                
                Double tmp0 = x * (Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, x / K)) - Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, x / B)));
                Double tmp1 = -Exp(-r * tau) * K * (Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, x / K)) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, x / B)));
                Double tmp2 = -B * Pow(x / B, -2.0 * r / Pow(sigma,2.0)) * (Phi(Call(Double{}, FactorPlus{}, sigma, r, tau, Pow(B, 2.0 / K / x))) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, B / x)));
                Double tmp3_inner = Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, Pow(B, 2.0 / K / x))) - Phi(Call(Double{}, FactorMinus{}, sigma, r, tau, B / x));
                Double tmp3 = Exp(-r * tau) * K * Pow(x / B, -2 * r / Pow(sigma,2.0) + 1) * tmp3_inner;
                Double pv = tmp2;
                return pv;
            }
            std::vector<std::string> Arguments()const
            {
                return { "x", "K", "tau", "r", "sigma", "B"};
            }
            std::string Name()const
            {
                return "__KoBarrierCallOption";
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
        };
    };


} // end namespace KoBarrierOption


struct MyMax {

    template<class Double>
    struct Build {
        Double Evaluate(
            Double x,
            Double y)const
        {
            using MathFunctions::Max;
            return Max(x, y);
        }
        std::vector<std::string> Arguments()const
        {
            return { "x", "y"};
        }
        std::string Name()const
        {
            return "MyMax";
        }
    };
};


struct MyLogKFDivStd{

    template<class Double>
    struct Build {
        Double Evaluate(
            Double K,
            Double F,
            Double std)const
        {
            using MathFunctions::Log;
            Double result = Log(F / K) / std;
            return result;
        }
        std::vector<std::string> Arguments()const
        {
            return { "K", "F", "std"};
        }
        std::string Name()const
        {
            return "MyLogKFDivStd";
        }

        Double EvaluateVec(std::vector<Double> const& args)
        {
            enum { NumArgs = 3 };
            if (args.size() != NumArgs)
            {
                throw std::runtime_error("bad number of args");
            }
            return Evaluate(
                args[0],
                args[1],
                args[2]);
        }
        template<class F>
        Double Invoke(F&& f, std::vector<Double> const& args)
        {
            enum { NumArgs = 3 };
            if (args.size() != NumArgs)
            {
                throw std::runtime_error("bad number of args");
            }
            return f(
                args[0],
                args[1],
                args[2]);
        }
    };
};


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
            using MathFunctions::If;
            using MathFunctions::Call;

            Double df = Exp(-r * T);
            Double F = S * Exp(r * T);
            Double std = vol * Pow(T, 0.5);
            //Double d = Log(F / K) / std;
            Double d = Call(Double{}, MyLogKFDivStd{}, K, F, std);
            Double d1 = d + 0.5 * std;
            Double d2 = d1 - std;
            Double nd1 = Phi(d1);
            Double nd2 = Phi(d2);
            Double c = df * (F * nd1 - K * nd2);
            return c;

#if 0
            return If(
                special_condition,
                [&]() {
                    Double df = Exp(-r * T);
                    Double F = S * Exp(r * T);
                    Double std = vol * Pow(T, 0.5);
                    //Double d = Log(F / K) / std;
                    Double d = Call(MyLogKFDivStd{}, K, F, std);
                    Double d1 = d + 0.5 * std;
                    Double d2 = d1 - std;
                    Double nd1 = Phi(d1);
                    Double nd2 = Phi(d2);
                    Double c = df * (F * nd1 - K * nd2);
                    return c;
                },
                [&]()
                {
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
                });
#endif
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


#if 0

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

    using aad_generater_ty = AADFunctionGeneratorEx<kernel_ty>;
    //ticker.emplace_back("BlackScholesThreeAddressAADFwd", std::make_shared<aad_generater_ty>(aad_generater_ty::AADPT_Forwards));
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
    
#endif
   
    

#include "BlackGenerated.h"

#include <chrono>
#include <string>

#if 0


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


std::shared_ptr<Module> BuildRecursiveEx(std::shared_ptr<Operator> const& head)
{
    struct CtrlFinder
    {
        void operator()(std::shared_ptr<Operator> const& op)
        {
            if (auto ptr = std::dynamic_pointer_cast<If>(op))
            {
                ctrl_stmts_.push_back(ptr);
            }
        }

        std::vector<std::shared_ptr<Operator> > ctrl_stmts_;
    };

    auto ctrl_finder = std::make_shared< CtrlFinder>();
    head->VisitTopDown(*ctrl_finder);

    if (ctrl_finder->ctrl_stmts_.size() > 0)
    {
        auto untyped_ctrl_stmt = ctrl_finder->ctrl_stmts_[0];
        if (auto if_stmt = std::dynamic_pointer_cast<If>(untyped_ctrl_stmt))
        {
            auto if_expr = if_stmt;

            // I expect this to be a block of variable declerations,
            // the last decleration will be the conditional variable
            auto cond = BuildRecursiveEx(if_expr->Cond());
            auto cond_module = std::dynamic_pointer_cast<Module>(cond);
            auto cond_ib = std::dynamic_pointer_cast<InstructionBlock>(cond_module->back());

            auto cond_var_name = [&]()->std::string
            {
                if (auto last_instr = std::dynamic_pointer_cast<InstructionReturn>(cond_ib->back()))
                {
                    auto var_name = last_instr->VarName();
                    cond_ib->pop_back();
                    return var_name;
                }
                throw std::runtime_error("unexpected");
            }();

            auto if_true = BuildRecursiveEx(if_expr->IfTrue());
            auto if_false = BuildRecursiveEx(if_expr->IfFalse());

            auto if_block = std::make_shared< IfBlock>(
                cond_var_name,
                if_true,
                if_false);

            auto modulee = std::make_shared<Module>();
            modulee->push_back(cond_ib);
            modulee->push_back(if_block);
            return modulee;
        }
        else
        {
            throw std::domain_error("unexpectded");
        }
    }
    else
    {
        std::unordered_set<std::string> symbols_seen;

        auto IB = std::make_shared<InstructionBlock>();

        auto deps = head->DepthFirstAnySymbolicDependencyAndThis();
        if (deps.DepthFirst.size() > 0)
        {
            for (auto sym : deps.DepthFirst) {
                if (sym->IsExo())
                    continue;
                if (symbols_seen.count(sym->Name()) != 0)
                {
                    continue;
                }
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(sym)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(sym->Name(), expr));
                symbols_seen.insert(sym->Name());

            }
        }

        std::string aux_name = "result";
        IB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, head));
        IB->Add(std::make_shared< InstructionReturn>(aux_name));


        auto modulee = std::make_shared<Module>();
        modulee->push_back(IB);
        return modulee;

    }
}

std::shared_ptr<Operator> ExpandCall(std::shared_ptr<Operator> const& head)
{
    struct MapCallSite : OperatorTransform {
        virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
            auto candidate = ptr->Clone(shared_from_this());
            if (auto as_call = std::dynamic_pointer_cast<Call>(candidate))
            {
                auto name = "__tmp_call_" + std::to_string((size_t)as_call.get());
                names_.push_back(name);
                auto exo = std::make_shared<EndgenousSymbol>(name, candidate);
                return exo;
            }
            return candidate;
        }
        std::vector<std::string> names_;
    };

    auto result = head->Clone(std::make_shared<MapCallSite>());
    return result;
}

struct DebugControlBlockVisitor : ControlBlockVisitor
{
    size_t indent_ = 0;
    void indent()
    {
        if (indent_ != 0)
        {
            std::cout << std::string(indent_ * 4, ' ');
        }
    }
    void AcceptInstruction(const std::shared_ptr<const Instruction>& instr) override
    {
        indent();
        std::cout << "AcceptInstruction : ";
        instr->EmitCode(std::cout);
        if (auto as_lvalue_assign = std::dynamic_pointer_cast<const InstructionDeclareVariable>(instr))
        {
            if (auto as_call = std::dynamic_pointer_cast<const Call>(as_lvalue_assign->as_operator_()))
            {
                std::cout << " // is a call site\n";
            }
            
        }
    }
    void AcceptIf(const std::shared_ptr<const IfBlock>& if_block)
    {
        indent();
        std::cout << "BEGIN IF BLOCK\n";
        ++indent_;
        if_block->IfTrue()->Accept(*this);
        --indent_;
        indent();
        std::cout << "BEGIN ELSE BLOCK\n";
        ++indent_;
        if_block->IfFalse()->Accept(*this);
        --indent_;
        indent();
        std::cout << "END IF BLOCK\n";

    }
    void AcceptCall(const std::shared_ptr<const CallBlock>& call_block)
    {
        indent();
        std::cout << "CALL SITE\n";
    }
};

namespace Cady
{
    namespace ProgramCode
    {
        class Type {};
        class DoubleType : Type {};

        struct RValue
        {
            virtual ~RValue() = default;
            virtual std::string ToString()const = 0;
            virtual std::shared_ptr<Operator> to_operator()const = 0;
        };

        struct DoubleConstant : public RValue
        {
            explicit DoubleConstant(double value) :value_{ value } {}
            double Value()const { return value_; }
            virtual std::string ToString()const override { return std::to_string(value_); }
            virtual std::shared_ptr<Operator> to_operator()const override
            {
                return std::make_shared<Constant>(value_);
            }
        private:
            double value_;
        };

        struct LValue : public RValue
        {
        public:
            LValue(std::string const& name)
                : name_{ name }
            {}
            std::string Name()const { return name_; }
            virtual std::string ToString()const override {
                return Name();
            }
            virtual std::shared_ptr<Operator> to_operator()const override
            {
                return std::make_shared<ExogenousSymbol>(name_);
            }
        private:
            std::string name_;
        };





        struct Statement
        {
            virtual ~Statement() = default;
        };

        struct StatementList : Statement, std::vector<std::shared_ptr< Statement> >
        {
            StatementList(std::vector<std::shared_ptr< Statement> > const& args):
                std::vector<std::shared_ptr< Statement> >{ args }
            {}
        };

        class Assignment : public Statement {};

        enum OpCode
        {

            // unary
            OP_ASSIGN,
            OP_SQRT,
            OP_ERFC,
            OP_USUB,
            OP_EXP,
            OP_LOG,
            OP_PHI,


            // binary
            OP_ADD,
            OP_SUB,
            OP_DIV,
            OP_MUL,
            OP_POW,
            OP_MIN,
            OP_MAX,

            
            
        };
        inline std::ostream& operator<<(std::ostream& ostr, OpCode op)
        {
            switch (op) {
            case OP_ASSIGN: return ostr << "OP_ASSIGN";
            case OP_SQRT: return ostr << "OP_SQRT";
            case OP_ERFC: return ostr << "OP_ERFC";
            case OP_USUB: return ostr << "OP_USUB";
            case OP_EXP: return ostr << "OP_EXP";
            case OP_LOG: return ostr << "OP_LOG";
            case OP_PHI: return ostr << "OP_PHI";
                

            case OP_ADD: return ostr << "OP_ADD";
            case OP_SUB: return ostr << "OP_SUB";
            case OP_MUL: return ostr << "OP_MUL";
            case OP_DIV: return ostr << "OP_DIV";
            case OP_POW: return ostr << "OP_POW";
            case OP_MIN: return ostr << "OP_MIN";
            case OP_MAX: return ostr << "OP_MAX";
            }
            return ostr << "unknown";
        }

        class ThreeAddressCode : public Assignment
        {
        public:
            ThreeAddressCode(
                OpCode op,
                std::shared_ptr<const LValue> name,
                std::shared_ptr<const RValue> l_param,
                std::shared_ptr<const RValue> r_param)
                : op_{ op }
                , name_{ name }
                , l_param_{ l_param }
                , r_param_{ r_param }
            {}

            std::shared_ptr<Operator> to_operator()const
            {
                switch (op_)
                {
                case OP_ADD:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_ADD,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_SUB:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_SUB,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_MUL:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_MUL,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_DIV:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_DIV,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_POW:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_POW,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_MIN:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_MIN,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                case OP_MAX:
                    return std::make_shared<BinaryOperator>(
                        BinaryOperatorKind::OP_MAX,
                        l_param_->to_operator(),
                        r_param_->to_operator());
                default:
                    throw std::domain_error("TODO");
                }
            }

            OpCode op_;
            std::shared_ptr<const LValue> name_;
            std::shared_ptr<const RValue> l_param_;
            std::shared_ptr<const RValue> r_param_;
        };

        struct TwoAddressCode : public Assignment
        {


            TwoAddressCode(
                OpCode op,
                std::shared_ptr<const LValue> rvalue,
                std::shared_ptr<const RValue> param)
                : op_{ op }, rvalue_ {
                rvalue
            }, param_{ param }
            {}

            std::shared_ptr<Operator> to_operator()const
            {
                switch (op_)
                {
                case OP_USUB:
                    return std::make_shared<UnaryOperator>(
                        UnaryOperatorKind::UOP_USUB,
                        param_->to_operator());
                case OP_PHI:
                    return std::make_shared<Phi>(
                        param_->to_operator());
                case OP_EXP:
                    return std::make_shared<Exp>(
                        param_->to_operator());
                case OP_LOG:
                    return std::make_shared<Log>(
                        param_->to_operator());
                case OP_ASSIGN:
                    return param_->to_operator();
                default:
                    throw std::domain_error("TODO");
                }
            }

            OpCode op_;
            std::shared_ptr<const LValue> rvalue_;
            std::shared_ptr<const RValue> param_;
        };


        class IfStatement : public Statement
        {
        public:
            IfStatement(
                std::shared_ptr<RValue> const& condition,
                std::shared_ptr<Statement> const& if_true,
                std::shared_ptr<Statement> const& if_false)
                : condition_{ condition }
                , if_true_{ if_true }
                , if_false_{if_false}
            {}
 
            std::shared_ptr<RValue> condition_;
            std::shared_ptr<Statement> if_true_;
            std::shared_ptr<Statement> if_false_;
        };

        class WhileStatement : public Statement
        {
            std::shared_ptr<RValue> cond_;
            std::shared_ptr<Statement> stmt_;
        };


        struct CallStatement : public Statement
        {
            CallStatement(
                std::string function_name,
                std::vector<std::shared_ptr<LValue> > result_list,
                std::vector<std::shared_ptr<RValue> > arg_list)
                : function_name_{function_name}
                , result_list_{ result_list }
                , arg_list_{ arg_list }
            {}

            std::string function_name_;
            std::vector<std::shared_ptr<LValue> > result_list_;
            std::vector<std::shared_ptr<RValue> > arg_list_;
        };

        struct ReturnStatement : public Statement
        {
            explicit ReturnStatement(std::shared_ptr<RValue> const& value) :value_{ value } {}

            std::shared_ptr<RValue> value_;
        };

        struct ReturnArrayStatement : public Statement
        {
            explicit ReturnArrayStatement(
                std::vector<std::shared_ptr<RValue> > const& value_list)
                : value_list_{ value_list }
            {
            }

            std::vector<std::shared_ptr<RValue> > value_list_;
        };


        struct Function
        {
            Function(
                std::string const& name,
                std::vector<std::string> const& args, 
                std::vector<std::shared_ptr<Statement> > const& stmts)
                : name_{ name }, args_ {
                args
            }, stmts_{ std::make_shared<StatementList>(stmts)
            }
            {}

            void DebugPrintStmt(std::shared_ptr<Statement> const& stmt, size_t indent)const
            {
                if (auto three_addr = std::dynamic_pointer_cast<ThreeAddressCode>(stmt))
                {
                    std::cout << three_addr->name_->ToString() << " = "
                        << three_addr->l_param_->ToString() << " "
                        << three_addr->op_ << " "
                        << three_addr->r_param_->ToString() << ";\n";

                }
                else if (auto two_addr = std::dynamic_pointer_cast<TwoAddressCode>(stmt))
                {
                    std::cout << two_addr->rvalue_->ToString() << " = "
                        << two_addr->op_ << " "
                        << two_addr->param_->ToString() << ";\n";

                }
                else if (auto if_stmt = std::dynamic_pointer_cast<IfStatement>(stmt))
                {
                    std::cout << "if( !! " << if_stmt->condition_->ToString() << " ){\n";
                    DebugPrintStmt(if_stmt->if_true_, indent + 1);
                    std::cout << "} else {\n";
                    DebugPrintStmt(if_stmt->if_false_, indent + 1);
                    std::cout << "}\n";
                }
                else if (auto stmts = std::dynamic_pointer_cast<StatementList>(stmt))
                {
                    for (auto const& x : *stmts)
                    {
                        DebugPrintStmt(x, indent);
                    }
                }
                else if (auto return_stmt = std::dynamic_pointer_cast<ReturnStatement>(stmt))
                {
                    std::cout << "return " << return_stmt->value_->ToString() << "\n";
                }
                else if (auto call_stmt = std::dynamic_pointer_cast<CallStatement>(stmt))
                {
                    std::cout << "CALL " << call_stmt->function_name_ << "(...)\n";
                }
                else
                {
                    std::string();
                }
            }

            void DebugPrint()const
            {
                for (auto const& stmt : *stmts_)
                {
                    DebugPrintStmt(stmt, 0);
                }
            }
            auto const& Name()const { return name_; }
            auto const& Args()const { return args_; }
            auto const& Statements()const { return stmts_;  }
            auto& Statements(){ return stmts_; }
        private:
            std::string name_;
            std::vector<std::string> args_;
            std::shared_ptr<StatementList> stmts_;
        };

        struct Namespace
        {
            Namespace& AddFunction(std::shared_ptr<Function> const& ptr)
            {
                functions_.push_back(ptr);
                return *this;
            }
        private:
            std::vector<std::shared_ptr<Function> > functions_;
        };


        struct CodeWriter
        {
            void EmitCode(std::ostream& ostr, std::shared_ptr< Function> const& f)const
            {
                ostr << "auto " << f->Name() << "(";
                auto const& args = f->Args();
                for (size_t idx = 0; idx != args.size(); ++idx)
                {
                    ostr << (idx == 0 ? "" : ", ") << "const double " << args[idx];
                }
                ostr << ")\n";
                ostr << "{\n";
                EmitCodeForStatement(ostr, f->Statements(), 1);
                ostr << "}\n";
            }
            void EmitCodeForStatement(std::ostream& ostr, std::shared_ptr<Statement> const& stmt, size_t indent)const
            {
                auto do_indent = [&]() {
                    if (indent != 0)
                    {
                        ostr << std::string(indent * 4, ' ');
                    }
                };
                if (auto three_addr = std::dynamic_pointer_cast<ThreeAddressCode>(stmt))
                {
                    do_indent();
                    ostr << "const double " << three_addr->name_->ToString() << " = ";
                    switch (three_addr->op_)
                    {
                    case OP_ADD:
                        ostr << three_addr->l_param_->ToString() << " + " << three_addr->r_param_->ToString() << ";\n";
                        break;
                    case OP_SUB:
                        ostr << three_addr->l_param_->ToString() << " - " << three_addr->r_param_->ToString() << ";\n";
                        break;
                    case OP_MUL:
                        ostr << three_addr->l_param_->ToString() << " * " << three_addr->r_param_->ToString() << ";\n";
                        break;
                    case OP_DIV:
                        ostr << three_addr->l_param_->ToString() << " / " << three_addr->r_param_->ToString() << ";\n";
                        break;
                    case OP_POW:
                        ostr << "std::pow(" << three_addr->l_param_->ToString() << "," << three_addr->r_param_->ToString() << ");\n";
                        break;
                    case OP_MIN:
                        ostr << "std::min(" << three_addr->l_param_->ToString() << "," << three_addr->r_param_->ToString() << ");\n";
                        break;
                    case OP_MAX:
                        ostr << "std::max(" << three_addr->l_param_->ToString() << "," << three_addr->r_param_->ToString() << ");\n";
                        break;
                    default:
                        throw std::domain_error("TODO");
                    }
                }
                else if (auto two_addr = std::dynamic_pointer_cast<TwoAddressCode>(stmt))
                {
                    do_indent();
                    ostr << "const double " << two_addr->rvalue_->ToString() << " = ";
                    switch (two_addr->op_)
                    {
                    case OP_USUB:
                        ostr << "- " << two_addr->param_->ToString() << ";\n";
                        break;
                    case OP_ASSIGN:
                        ostr << two_addr->param_->ToString() << ";\n";
                        break;
                    case OP_EXP:
                        ostr << "std::exp(" << two_addr->param_->ToString() << ");\n";
                        break;
                    case OP_LOG:
                        ostr << "std::log(" << two_addr->param_->ToString() << ");\n";
                        break;
                    case OP_PHI:
                        ostr << "std::erfc(-(";
                        ostr << two_addr->param_->ToString();
                        ostr << ")/std::sqrt(2))/2;\n";
                        break;
                    default:
                        throw std::domain_error("TODO");
                    }

                }
                else if (auto if_stmt = std::dynamic_pointer_cast<IfStatement>(stmt))
                {
                    do_indent();
                    ostr << "if( !! " << if_stmt->condition_->ToString() << " ){\n";
                    EmitCodeForStatement(ostr,if_stmt->if_true_, indent + 1);
                    do_indent();
                    std::cout << "} else {\n";
                    EmitCodeForStatement(ostr, if_stmt->if_false_, indent + 1);
                    do_indent();
                    std::cout << "}\n";
                }
                else if (auto stmts = std::dynamic_pointer_cast<StatementList>(stmt))
                {
                    for (auto const& x : *stmts)
                    {
                        EmitCodeForStatement(ostr, x, indent);
                    }
                }
                else if (auto return_stmt = std::dynamic_pointer_cast<ReturnStatement>(stmt))
                {
                    do_indent();
                    ostr << "return " << return_stmt->value_->ToString() << ";\n";
                }
                else if (auto return_stmt = std::dynamic_pointer_cast<ReturnArrayStatement>(stmt))
                {
                    do_indent();
                    ostr << "return std::array<double, " << return_stmt->value_list_.size() << ">{";
                    for (size_t idx = 0; idx != return_stmt->value_list_.size(); ++idx)
                    {
                        ostr << (idx == 0 ? "" : ", ") << return_stmt->value_list_[idx]->ToString();
                    }
                    ostr << "};\n";
                }
                else if (auto call_stmt = std::dynamic_pointer_cast<CallStatement>(stmt))
                {
                    do_indent();
                    std::string call_result_token = "__call_result_" + std::to_string((std::size_t)call_stmt.get());
                    ostr << "auto " << call_result_token << " = " << call_stmt->function_name_ << "(";
                    for (size_t idx = 0; idx != call_stmt->arg_list_.size(); ++idx)
                    {
                        ostr << (idx == 0 ? "" : ", ") << call_stmt->arg_list_[idx]->ToString();
                    }
                    ostr << ");\n";
                    for(size_t idx=0;idx!=call_stmt->result_list_.size();++idx)
                    {
                        auto const& lvalue = call_stmt->result_list_[idx];
                        do_indent();
                        ostr << "const double " << lvalue->ToString() << " = " << call_result_token << "[" << idx << "];\n";
                    }
                }
                else
                {
                    std::string();
                }
            }
        };

        

    } // end namespace ProgramCode

} // end namespace Cady


/*
Creates a linear sequence of instructions, and splits out if statement, and call sites
*/
struct InstructionLinearizer : ControlBlockVisitor
{
    void AcceptInstruction(const std::shared_ptr<const Instruction>& instr) override
    {

        if (auto as_lvalue_assign = std::dynamic_pointer_cast<const InstructionDeclareVariable>(instr))
        {
            if (seen_.count(instr))
            {
                // short circute
                return;
            }
            auto make_rvalue = [](std::shared_ptr<Operator> const& op)->std::shared_ptr<ProgramCode::RValue>
            {
                if (auto as_sym = std::dynamic_pointer_cast<const Symbol>(op))
                {
                    return std::dynamic_pointer_cast<ProgramCode::RValue>(std::make_shared<ProgramCode::LValue>(as_sym->Name()));
                }
                if (auto as_lit = std::dynamic_pointer_cast<const Constant>(op))
                {
                    return std::make_shared<ProgramCode::DoubleConstant>(as_lit->Value());
                }
                throw std::domain_error("not an rvalue");
            };

            if (auto as_call = std::dynamic_pointer_cast<const Call>(as_lvalue_assign->as_operator_()))
            {
                std::vector<std::shared_ptr<ProgramCode::RValue> > arg_list;
                for (auto const& arg : as_call->Children())
                {
                    arg_list.push_back(make_rvalue(arg));
                }
                auto call_stmt = std::make_shared<ProgramCode::CallStatement>(
                    as_call->FunctionName(),
                    std::vector<std::shared_ptr<ProgramCode::LValue> >{std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName())},
                    arg_list);
                stmts_.push_back(call_stmt);
            }
            else
            {
                auto op = as_lvalue_assign->as_operator_();
                if (auto as_binary = std::dynamic_pointer_cast<const BinaryOperator>(op))
                {
                    auto left_name = make_rvalue(as_binary->LParam());
                    auto right_name = make_rvalue(as_binary->RParam());

                    auto mapped_op = [&]()->ProgramCode::OpCode
                    {
                        using ProgramCode::OpCode;
                        switch (as_binary->OpKind())
                        {
                        case BinaryOperatorKind::OP_ADD: return OpCode::OP_ADD;
                        case BinaryOperatorKind::OP_SUB: return OpCode::OP_SUB;
                        case BinaryOperatorKind::OP_MUL: return OpCode::OP_MUL;
                        case BinaryOperatorKind::OP_DIV: return OpCode::OP_DIV;
                        case BinaryOperatorKind::OP_POW: return OpCode::OP_POW;
                        case BinaryOperatorKind::OP_MIN: return OpCode::OP_MIN;
                        case BinaryOperatorKind::OP_MAX: return OpCode::OP_MAX;
                        }
                        throw std::domain_error("unknown binary op");
                    }();

                    auto three_address = std::make_shared<ProgramCode::ThreeAddressCode>(
                        mapped_op,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        left_name,
                        right_name);

                    stmts_.push_back(three_address);

                }
                else if (auto as_unary = std::dynamic_pointer_cast<const UnaryOperator>(op))
                {
                    auto mapped_op = [&]()->ProgramCode::OpCode
                    {
                        using ProgramCode::OpCode;
                        switch (as_unary->OpKind())
                        {
                        case UnaryOperatorKind::UOP_USUB: return OpCode::OP_USUB;
                        }
                        throw std::domain_error("unknown unaru op");
                    }();

                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        mapped_op,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        make_rvalue(as_unary->At(0)));

                    stmts_.push_back(two_address);
                }
                else if (auto as_sym = std::dynamic_pointer_cast<const Symbol>(op))
                {
                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        ProgramCode::OpCode::OP_ASSIGN,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        std::make_shared<ProgramCode::LValue>(as_sym->Name()));

                    stmts_.push_back(two_address);
                }
                else if (auto as_exp = std::dynamic_pointer_cast<const Exp>(op))
                {
                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        ProgramCode::OpCode::OP_EXP,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        make_rvalue(as_exp->At(0)));

                    stmts_.push_back(two_address);
                }
                else if (auto as_log = std::dynamic_pointer_cast<const Log>(op))
                {
                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        ProgramCode::OpCode::OP_LOG,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        make_rvalue(as_log->At(0)));

                    stmts_.push_back(two_address);
                }
                else if (auto as_phi = std::dynamic_pointer_cast<const Phi>(op))
                {
                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        ProgramCode::OpCode::OP_PHI,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        make_rvalue(as_phi->At(0)));

                    stmts_.push_back(two_address);
                }
                else if (auto as_constant = std::dynamic_pointer_cast<const Constant>(op))
                {
                    auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                        ProgramCode::OpCode::OP_ASSIGN,
                        std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                        std::make_shared<ProgramCode::DoubleConstant>(as_constant->Value()));

                    stmts_.push_back(two_address);
                }
                else
                {
                    std::string();
                }
            }
        }
        else if (auto as_return = std::dynamic_pointer_cast<const InstructionReturn>(instr))
        {
            stmts_.push_back(std::make_shared<ProgramCode::ReturnStatement>(
                std::make_shared<ProgramCode::LValue>(as_return->VarName())));
        }
        else
        {
            throw std::domain_error("unknown type");
        }

        seen_.insert(instr);
    }
    void AcceptIf(const std::shared_ptr<const IfBlock>& if_block)
    {
        InstructionLinearizer if_true;
        InstructionLinearizer if_false;
        if_block->IfTrue()->Accept(if_true);
        if_block->IfFalse()->Accept(if_false);
        
        auto result = std::make_shared<ProgramCode::IfStatement>(
            std::make_shared<ProgramCode::LValue>(if_block->ConditionVariable()),
            std::make_shared<ProgramCode::StatementList>(if_true.stmts_),
            std::make_shared<ProgramCode::StatementList>(if_false.stmts_));
        stmts_.push_back(result);
    }
    void AcceptCall(const std::shared_ptr<const CallBlock>& call_block)
    {

    }
    std::unordered_set<std::shared_ptr<const Instruction> > seen_;
    std::vector<std::shared_ptr<ProgramCode::Statement> > stmts_;
};


struct ExecutionContext
{
    std::vector<std::string> exo_names;
    std::vector<std::unordered_map<std::string, size_t> > alloc_map_list;
    std::vector<std::shared_ptr<ImpliedMatrixFunction> > jacobian_list;
};


std::shared_ptr<ProgramCode::Statement> CloneStmtWithDiffs(
    ExecutionContext& context,
    std::shared_ptr<ProgramCode::Statement> const& stmt)
{

    using namespace ProgramCode;

    if (auto stmts = std::dynamic_pointer_cast<StatementList>(stmt))
    {
        std::vector<std::shared_ptr<Statement> > new_stmts;
        for (auto const& child_stmt : *stmts)
        {
            auto result = CloneStmtWithDiffs(
                context,
                child_stmt);
            new_stmts.push_back(result);
            
        }
        return std::make_shared<StatementList>(std::vector<std::shared_ptr<Statement> >{new_stmts});
    }
    else if (auto three_addr = std::dynamic_pointer_cast<ThreeAddressCode>(stmt))
    {
        std::string lvalue = three_addr->name_->ToString();
        auto alloc_map = context.alloc_map_list.back();
        if (alloc_map.count(lvalue) == 0)
        {
            auto expr = three_addr->to_operator();


            auto slot = alloc_map.size();
            alloc_map[lvalue] = slot;
            
            auto matrix_func = ImpliedMatrixFunction::MakeFromOperator(context.jacobian_list.size(), expr);
            context.alloc_map_list.push_back(alloc_map);
            context.jacobian_list.push_back(matrix_func);
        }
        return stmt;
    }
    else if (auto two_addr = std::dynamic_pointer_cast<TwoAddressCode>(stmt))
    {
        auto lvalue = two_addr->rvalue_->ToString();
        auto alloc_map = context.alloc_map_list.back();
        if (alloc_map.count(lvalue) == 0)
        {
            auto expr = two_addr->to_operator();
            auto slot = alloc_map.size();
            alloc_map[lvalue] = slot;
            if (alloc_map.size() != slot + 1)
            {
                std::string();
            }
            

            auto matrix_func = ImpliedMatrixFunction::MakeFromOperator(context.jacobian_list.size(), expr);
            context.alloc_map_list.push_back(alloc_map); 
            context.jacobian_list.push_back(matrix_func);
        }
        return stmt;
    }
    else if (auto if_stmt = std::dynamic_pointer_cast<IfStatement>(stmt))
    {
        ExecutionContext true_context(context);
        ExecutionContext false_context(context);
        auto true_stmt = CloneStmtWithDiffs(true_context, if_stmt->if_true_);
        auto false_stmt = CloneStmtWithDiffs(true_context, if_stmt->if_false_);
        return std::make_shared<IfStatement>(
            if_stmt->condition_, true_stmt, false_stmt);

    }
    else if (auto call_stmt = std::dynamic_pointer_cast<CallStatement>(stmt))
    {
        auto mapped_result_list = call_stmt->result_list_;
        if (mapped_result_list.size() != 1)
        {
            throw std::domain_error("unexpected");
        }
        for (auto const& arg : call_stmt->arg_list_)
        {
            std::string d_name = "d_" + std::to_string((size_t)call_stmt.get()) + "_" + arg->ToString();
            mapped_result_list.push_back(std::make_shared<LValue>(d_name));
        }
        
        

        const auto id = context.jacobian_list.size();
        std::vector<std::shared_ptr<Symbol> > args;
        std::vector<std::shared_ptr<Operator> > diffs;
        for (auto const& arg : call_stmt->arg_list_)
        {
            args.push_back(std::make_shared<ExogenousSymbol>(arg->ToString()));
        }
        for (size_t idx = 1; idx < mapped_result_list.size();++idx)
        {
            auto const& d = mapped_result_list[idx];
            diffs.push_back(std::make_shared<ExogenousSymbol>(d->ToString()));
        }
        auto jacoibian = std::make_shared< ImpliedMatrixFunction>(id, "dummy", args, diffs);

        auto lvalue = call_stmt->result_list_[0]->ToString();
        auto alloc_map = context.alloc_map_list.back();
        auto slot = alloc_map.size();
        alloc_map[lvalue] = slot;
        

        context.alloc_map_list.push_back(alloc_map);
        context.jacobian_list.push_back(jacoibian);

        return std::make_shared<CallStatement>(
            call_stmt->function_name_,
            mapped_result_list,
            call_stmt->arg_list_);
    }
    else if (auto return_stmt = std::dynamic_pointer_cast<ReturnStatement>(stmt))
    {
        // HERE we add the the jacobian


#if 0
        for (auto const& func : context.jacobian_list)
        {
            func->PrintDebug();
        }
#endif

        std::vector<std::shared_ptr<SymbolicMatrix> > adj_matrix_list;
        for (size_t idx = context.jacobian_list.size(); idx != 0; )
        {
            bool is_terminal = (idx == context.jacobian_list.size());
            --idx;

            auto const& alloc_map = context.alloc_map_list[idx];

            auto matrix_decl = context.jacobian_list[idx]->MakeMatrix(alloc_map, is_terminal);

            adj_matrix_list.push_back(matrix_decl->Matrix());

        }

        auto fold_backwards = [&]()
        {
            // (M1*(M2*M3))
            std::shared_ptr<SymbolicMatrix> adj_matrix = adj_matrix_list[0];
            for (size_t idx = 1; idx < adj_matrix_list.size(); ++idx)
            {
                adj_matrix = adj_matrix_list[idx]->Multiply(*adj_matrix);
            }
            return adj_matrix;
        };

        auto adj_matrix = fold_backwards();

        auto three_address_transform = std::make_shared<Transform::RemapUnique>("__adj");

        std::unordered_set<std::string> three_addr_seen;
#if 0
        for (auto const& p : context.alloc_map_list.back())
        {
            three_addr_seen.insert(p.first);
        }
#endif

        auto AADIB = std::make_shared<InstructionBlock>();

        std::vector<std::shared_ptr<RValue> > output_list;
        output_list.push_back(return_stmt->value_);


        for (auto const& exo : context.exo_names)
        {
            auto d_sym = std::string("d_") + exo;

            auto slot = context.alloc_map_list[0].find(exo)->second;
            auto d_expr_orig = adj_matrix->At(slot, 0);

            Transform::FoldZero fold_zero;
            auto d_expr_no_zero = fold_zero.Fold(d_expr_orig);
            auto d_expr_three_address = d_expr_no_zero->Clone(three_address_transform);

            auto deps = d_expr_three_address->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) == 0) {
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                    AADIB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                    three_addr_seen.insert(head->Name());
                }

            }

            std::string aux_name = "result_" + d_sym;
            output_list.push_back(std::make_shared<LValue>(aux_name));
            AADIB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, d_expr_three_address));


        }

        auto l = std::make_shared<InstructionLinearizer>();
        AADIB->Accept(*l);

        auto stmts_with_aad = l->stmts_;
        stmts_with_aad.push_back(std::make_shared<ReturnArrayStatement>(output_list));
        return std::make_shared<StatementList>(stmts_with_aad);
    }
    else
    {
        throw std::domain_error("todo");
    }

 }


std::shared_ptr<ProgramCode::Function> CloneWithDiffs(std::shared_ptr<ProgramCode::Function>& f)
{
    auto stmts = f->Statements();
    std::unordered_map<std::string, size_t> alloc_map;
    for (auto const& arg : f->Args())
    {
        auto slot = alloc_map.size();
        alloc_map[arg] = slot;
    }
    ExecutionContext context;
    context.exo_names = f->Args();
    context.alloc_map_list.push_back(alloc_map);
  
    auto mapped_stmts = CloneStmtWithDiffs(context, stmts);
    return std::make_shared<ProgramCode::Function>(
        f->Name(),
        f->Args(),
        std::vector< std::shared_ptr<ProgramCode::Statement>>{ mapped_stmts });
}


auto black(const double t, const double T, const double r, const double S, const double K, const double vol)
{
    const double __symbol_2 = T;
    const double __symbol_57 = t;
    const double __symbol_58 = __symbol_2 - __symbol_57;
    const double __statement_0 = __symbol_58;
    const double result = __statement_0;
    if (!!result) {
        const double __symbol_9 = r;
        const double __symbol_30 = -__symbol_9;
        const double __symbol_2 = T;
        const double __symbol_31 = __symbol_30 * __symbol_2;
        const double __symbol_32 = std::exp(__symbol_31);
        const double __statement_10 = __symbol_32;
        const double __symbol_12 = S;
        const double __symbol_10 = __symbol_9 * __symbol_2;
        const double __symbol_11 = std::exp(__symbol_10);
        const double __symbol_13 = __symbol_12 * __symbol_11;
        const double __statement_11 = __symbol_13;
        const double __symbol_8 = K;
        const double __symbol_39 = __statement_11 / __symbol_8;
        const double __symbol_40 = std::log(__symbol_39);
        const double __symbol_4 = vol;
        const double __symbol_3 = std::pow(__symbol_2, 0.500000);
        const double __symbol_5 = __symbol_4 * __symbol_3;
        const double __statement_12 = __symbol_5;
        const double __symbol_41 = __symbol_40 / __statement_12;
        const double __statement_13 = __symbol_41;
        const double __symbol_37 = 0.500000 * __statement_12;
        const double __symbol_43 = __statement_13 + __symbol_37;
        const double __statement_14 = __symbol_43;
        const double __symbol_50 = std::erfc(-(__statement_14) / std::sqrt(2)) / 2;
        const double __statement_16 = __symbol_50;
        const double __symbol_52 = __statement_11 * __statement_16;
        const double __symbol_45 = __statement_14 - __statement_12;
        const double __statement_15 = __symbol_45;
        const double __symbol_47 = std::erfc(-(__statement_15) / std::sqrt(2)) / 2;
        const double __statement_17 = __symbol_47;
        const double __symbol_49 = __symbol_8 * __statement_17;
        const double __symbol_53 = __symbol_52 - __symbol_49;
        const double __symbol_55 = __statement_10 * __symbol_53;
        const double __statement_18 = __symbol_55;
        const double result = __statement_18;
        const double result_d_t = 0.000000;
        const double __adj64 = __symbol_2;
        const double __adj65 = std::pow(__adj64, -0.500000);
        const double __adj66 = 0.500000 * __adj65;
        const double __adj61 = __symbol_4;
        const double __adj24 = __statement_15;
        const double __adj25 = std::pow(__adj24, 2.000000);
        const double __adj26 = 0.500000 * __adj25;
        const double __adj27 = -__adj26;
        const double __adj28 = std::exp(__adj27);
        const double __adj29 = __adj28 / 2.506628;
        const double __adj22 = __symbol_8;
        const double __adj7 = __statement_10;
        const double __adj21 = -1.000000 * __adj7;
        const double __adj23 = __adj22 * __adj21;
        const double __adj30 = __adj29 * __adj23;
        const double __adj58 = -1.000000 * __adj30;
        const double __adj12 = __statement_14;
        const double __adj13 = std::pow(__adj12, 2.000000);
        const double __adj15 = 0.500000 * __adj13;
        const double __adj16 = -__adj15;
        const double __adj17 = std::exp(__adj16);
        const double __adj18 = __adj17 / 2.506628;
        const double __adj8 = __statement_11;
        const double __adj9 = __adj8 * __adj7;
        const double __adj19 = __adj18 * __adj9;
        const double __adj31 = __adj30 + __adj19;
        const double __adj57 = 0.500000 * __adj31;
        const double __adj59 = __adj58 + __adj57;
        const double __adj53 = __symbol_40;
        const double __adj54 = -__adj53;
        const double __adj32 = __statement_12;
        const double __adj33 = std::pow(__adj32, 2.000000);
        const double __adj55 = __adj54 / __adj33;
        const double __adj56 = __adj55 * __adj31;
        const double __adj60 = __adj59 + __adj56;
        const double __adj62 = __adj61 * __adj60;
        const double __adj67 = __adj66 * __adj62;
        const double __adj51 = __symbol_9;
        const double __adj48 = __symbol_10;
        const double __adj49 = std::exp(__adj48);
        const double __adj46 = __symbol_12;
        const double __adj43 = __statement_16;
        const double __adj44 = __adj43 * __adj7;
        const double __adj40 = std::pow(__adj22, 2.000000);
        const double __adj41 = __adj22 / __adj40;
        const double __adj36 = __symbol_39;
        const double __adj38 = 1.000000 / __adj36;
        const double __adj34 = __adj32 / __adj33;
        const double __adj35 = __adj34 * __adj31;
        const double __adj39 = __adj38 * __adj35;
        const double __adj42 = __adj41 * __adj39;
        const double __adj45 = __adj44 + __adj42;
        const double __adj47 = __adj46 * __adj45;
        const double __adj50 = __adj49 * __adj47;
        const double __adj52 = __adj51 * __adj50;
        const double __adj68 = __adj67 + __adj52;
        const double __adj5 = __symbol_30;
        const double __adj2 = __symbol_31;
        const double __adj3 = std::exp(__adj2);
        const double __adj1 = __symbol_53;
        const double __adj4 = __adj3 * __adj1;
        const double __adj6 = __adj5 * __adj4;
        const double result_d_T = __adj68 + __adj6;
        const double __adj72 = __adj64 * __adj50;
        const double __adj70 = -1.000000;
        const double __adj69 = __adj64 * __adj4;
        const double __adj71 = __adj70 * __adj69;
        const double result_d_r = __adj72 + __adj71;
        const double __adj73 = __symbol_11;
        const double result_d_S = __adj73 * __adj45;
        const double __adj77 = __statement_17;
        const double __adj78 = __adj77 * __adj21;
        const double __adj74 = -__adj8;
        const double __adj75 = __adj74 / __adj40;
        const double __adj76 = __adj75 * __adj39;
        const double result_d_K = __adj78 + __adj76;
        const double __adj79 = __symbol_3;
        const double result_d_vol = __adj79 * __adj60;
        return std::array<double, 7>{result, result_d_t, result_d_T, result_d_r, result_d_S, result_d_K, result_d_vol};
    }
    else {
        const double __symbol_9 = r;
        const double __symbol_30 = -__symbol_9;
        const double __symbol_2 = T;
        const double __symbol_31 = __symbol_30 * __symbol_2;
        const double __symbol_32 = std::exp(__symbol_31);
        const double __statement_1 = __symbol_32;
        const double __symbol_12 = S;
        const double __symbol_10 = __symbol_9 * __symbol_2;
        const double __symbol_11 = std::exp(__symbol_10);
        const double __symbol_13 = __symbol_12 * __symbol_11;
        const double __statement_2 = __symbol_13;
        const double __symbol_8 = K;
        const double __symbol_15 = __statement_2 / __symbol_8;
        const double __symbol_16 = std::log(__symbol_15);
        const double __symbol_4 = vol;
        const double __symbol_3 = std::pow(__symbol_2, 0.500000);
        const double __symbol_5 = __symbol_4 * __symbol_3;
        const double __statement_3 = __symbol_5;
        const double __symbol_17 = __symbol_16 / __statement_3;
        const double __statement_4 = __symbol_17;
        const double __symbol_7 = 0.500000 * __statement_3;
        const double __symbol_19 = __statement_4 + __symbol_7;
        const double __statement_5 = __symbol_19;
        const double __symbol_26 = std::erfc(-(__statement_5) / std::sqrt(2)) / 2;
        const double __statement_7 = __symbol_26;
        const double __symbol_28 = __statement_2 * __statement_7;
        const double __symbol_21 = __statement_5 - __statement_3;
        const double __statement_6 = __symbol_21;
        const double __symbol_23 = std::erfc(-(__statement_6) / std::sqrt(2)) / 2;
        const double __statement_8 = __symbol_23;
        const double __symbol_25 = __symbol_8 * __statement_8;
        const double __symbol_29 = __symbol_28 - __symbol_25;
        const double __symbol_34 = __statement_1 * __symbol_29;
        const double __statement_9 = __symbol_34;
        const double result = __statement_9;
        const double result_d_t = 0.000000;
        const double __adj64 = __symbol_2;
        const double __adj65 = std::pow(__adj64, -0.500000);
        const double __adj66 = 0.500000 * __adj65;
        const double __adj61 = __symbol_4;
        const double __adj24 = __statement_6;
        const double __adj25 = std::pow(__adj24, 2.000000);
        const double __adj26 = 0.500000 * __adj25;
        const double __adj27 = -__adj26;
        const double __adj28 = std::exp(__adj27);
        const double __adj29 = __adj28 / 2.506628;
        const double __adj22 = __symbol_8;
        const double __adj7 = __statement_1;
        const double __adj21 = -1.000000 * __adj7;
        const double __adj23 = __adj22 * __adj21;
        const double __adj30 = __adj29 * __adj23;
        const double __adj58 = -1.000000 * __adj30;
        const double __adj12 = __statement_5;
        const double __adj13 = std::pow(__adj12, 2.000000);
        const double __adj15 = 0.500000 * __adj13;
        const double __adj16 = -__adj15;
        const double __adj17 = std::exp(__adj16);
        const double __adj18 = __adj17 / 2.506628;
        const double __adj8 = __statement_2;
        const double __adj9 = __adj8 * __adj7;
        const double __adj19 = __adj18 * __adj9;
        const double __adj31 = __adj30 + __adj19;
        const double __adj57 = 0.500000 * __adj31;
        const double __adj59 = __adj58 + __adj57;
        const double __adj53 = __symbol_16;
        const double __adj54 = -__adj53;
        const double __adj32 = __statement_3;
        const double __adj33 = std::pow(__adj32, 2.000000);
        const double __adj55 = __adj54 / __adj33;
        const double __adj56 = __adj55 * __adj31;
        const double __adj60 = __adj59 + __adj56;
        const double __adj62 = __adj61 * __adj60;
        const double __adj67 = __adj66 * __adj62;
        const double __adj51 = __symbol_9;
        const double __adj48 = __symbol_10;
        const double __adj49 = std::exp(__adj48);
        const double __adj46 = __symbol_12;
        const double __adj43 = __statement_7;
        const double __adj44 = __adj43 * __adj7;
        const double __adj40 = std::pow(__adj22, 2.000000);
        const double __adj41 = __adj22 / __adj40;
        const double __adj36 = __symbol_15;
        const double __adj38 = 1.000000 / __adj36;
        const double __adj34 = __adj32 / __adj33;
        const double __adj35 = __adj34 * __adj31;
        const double __adj39 = __adj38 * __adj35;
        const double __adj42 = __adj41 * __adj39;
        const double __adj45 = __adj44 + __adj42;
        const double __adj47 = __adj46 * __adj45;
        const double __adj50 = __adj49 * __adj47;
        const double __adj52 = __adj51 * __adj50;
        const double __adj68 = __adj67 + __adj52;
        const double __adj5 = __symbol_30;
        const double __adj2 = __symbol_31;
        const double __adj3 = std::exp(__adj2);
        const double __adj1 = __symbol_29;
        const double __adj4 = __adj3 * __adj1;
        const double __adj6 = __adj5 * __adj4;
        const double result_d_T = __adj68 + __adj6;
        const double __adj72 = __adj64 * __adj50;
        const double __adj70 = -1.000000;
        const double __adj69 = __adj64 * __adj4;
        const double __adj71 = __adj70 * __adj69;
        const double result_d_r = __adj72 + __adj71;
        const double __adj73 = __symbol_11;
        const double result_d_S = __adj73 * __adj45;
        const double __adj77 = __statement_8;
        const double __adj78 = __adj77 * __adj21;
        const double __adj74 = -__adj8;
        const double __adj75 = __adj74 / __adj40;
        const double __adj76 = __adj75 * __adj39;
        const double result_d_K = __adj78 + __adj76;
        const double __adj79 = __symbol_3;
        const double result_d_vol = __adj79 * __adj60;
        return std::array<double, 7>{result, result_d_t, result_d_T, result_d_r, result_d_S, result_d_K, result_d_vol};
    }
}

auto __MyLogKFDivStd(const double K, const double F, const double std)
{
    const double __symbol_3 = F;
    const double __symbol_2 = K;
    const double __symbol_4 = __symbol_3 / __symbol_2;
    const double __symbol_5 = std::log(__symbol_4);
    const double __symbol_1 = std;
    const double __symbol_6 = __symbol_5 / __symbol_1;
    const double __statement_0 = __symbol_6;
    const double result = __statement_0;
    const double __adj11 = __symbol_3;
    const double __adj12 = -__adj11;
    const double __adj9 = __symbol_2;
    const double __adj10 = std::pow(__adj9, 2.000000);
    const double __adj13 = __adj12 / __adj10;
    const double __adj5 = __symbol_4;
    const double __adj7 = 1.000000 / __adj5;
    const double __adj2 = __symbol_1;
    const double __adj3 = std::pow(__adj2, 2.000000);
    const double __adj4 = __adj2 / __adj3;
    const double __adj8 = __adj7 * __adj4;
    const double result_d_K = __adj13 * __adj8;
    const double __adj14 = __adj9 / __adj10;
    const double result_d_F = __adj14 * __adj8;
    const double __adj15 = __symbol_5;
    const double __adj16 = -__adj15;
    const double result_d_std = __adj16 / __adj3;
    return std::array<double, 4>{result, result_d_K, result_d_F, result_d_std};
}
auto __black(const double t, const double T, const double r, const double S, const double K, const double vol)
{
    const double __symbol_9 = r;
    const double __symbol_28 = -__symbol_9;
    const double __symbol_2 = T;
    const double __symbol_29 = __symbol_28 * __symbol_2;
    const double __symbol_30 = std::exp(__symbol_29);
    const double __statement_0 = __symbol_30;
    const double __symbol_12 = S;
    const double __symbol_10 = __symbol_9 * __symbol_2;
    const double __symbol_11 = std::exp(__symbol_10);
    const double __symbol_13 = __symbol_12 * __symbol_11;
    const double __statement_1 = __symbol_13;
    const double __symbol_8 = K;
    const double __symbol_4 = vol;
    const double __symbol_3 = std::pow(__symbol_2, 0.500000);
    const double __symbol_5 = __symbol_4 * __symbol_3;
    const double __statement_2 = __symbol_5;
    auto __call_result_22070196 = __MyLogKFDivStd(__symbol_8, __statement_1, __statement_2);
    const double __symbol_15 = __call_result_22070196[0];
    const double d___symbol_8 = __call_result_22070196[1];
    const double d___statement_1 = __call_result_22070196[2];
    const double d___statement_2 = __call_result_22070196[3];
    const double __statement_3 = __symbol_15;
    const double __symbol_7 = 0.500000 * __statement_2;
    const double __symbol_17 = __statement_3 + __symbol_7;
    const double __statement_4 = __symbol_17;
    const double __symbol_24 = std::erfc(-(__statement_4) / std::sqrt(2)) / 2;
    const double __statement_6 = __symbol_24;
    const double __symbol_26 = __statement_1 * __statement_6;
    const double __symbol_19 = __statement_4 - __statement_2;
    const double __statement_5 = __symbol_19;
    const double __symbol_21 = std::erfc(-(__statement_5) / std::sqrt(2)) / 2;
    const double __statement_7 = __symbol_21;
    const double __symbol_23 = __symbol_8 * __statement_7;
    const double __symbol_27 = __symbol_26 - __symbol_23;
    const double __symbol_32 = __statement_0 * __symbol_27;
    const double __statement_8 = __symbol_32;
    const double result = __statement_8;
    const double result_d_t = 0.000000;
    const double __adj53 = __symbol_2;
    const double __adj54 = std::pow(__adj53, -0.500000);
    const double __adj55 = 0.500000 * __adj54;
    const double __adj50 = __symbol_4;
    const double __adj24 = __statement_5;
    const double __adj25 = std::pow(__adj24, 2.000000);
    const double __adj26 = 0.500000 * __adj25;
    const double __adj27 = -__adj26;
    const double __adj28 = std::exp(__adj27);
    const double __adj29 = __adj28 / 2.506628;
    const double __adj22 = __symbol_8;
    const double __adj7 = __statement_0;
    const double __adj21 = -1.000000 * __adj7;
    const double __adj23 = __adj22 * __adj21;
    const double __adj30 = __adj29 * __adj23;
    const double __adj47 = -1.000000 * __adj30;
    const double __adj12 = __statement_4;
    const double __adj13 = std::pow(__adj12, 2.000000);
    const double __adj15 = 0.500000 * __adj13;
    const double __adj16 = -__adj15;
    const double __adj17 = std::exp(__adj16);
    const double __adj18 = __adj17 / 2.506628;
    const double __adj8 = __statement_1;
    const double __adj9 = __adj8 * __adj7;
    const double __adj19 = __adj18 * __adj9;
    const double __adj31 = __adj30 + __adj19;
    const double __adj46 = 0.500000 * __adj31;
    const double __adj48 = __adj47 + __adj46;
    const double __adj44 = d___statement_2;
    const double __adj45 = __adj44 * __adj31;
    const double __adj49 = __adj48 + __adj45;
    const double __adj51 = __adj50 * __adj49;
    const double __adj56 = __adj55 * __adj51;
    const double __adj42 = __symbol_9;
    const double __adj39 = __symbol_10;
    const double __adj40 = std::exp(__adj39);
    const double __adj37 = __symbol_12;
    const double __adj34 = __statement_6;
    const double __adj35 = __adj34 * __adj7;
    const double __adj32 = d___statement_1;
    const double __adj33 = __adj32 * __adj31;
    const double __adj36 = __adj35 + __adj33;
    const double __adj38 = __adj37 * __adj36;
    const double __adj41 = __adj40 * __adj38;
    const double __adj43 = __adj42 * __adj41;
    const double __adj57 = __adj56 + __adj43;
    const double __adj5 = __symbol_28;
    const double __adj2 = __symbol_29;
    const double __adj3 = std::exp(__adj2);
    const double __adj1 = __symbol_27;
    const double __adj4 = __adj3 * __adj1;
    const double __adj6 = __adj5 * __adj4;
    const double result_d_T = __adj57 + __adj6;
    const double __adj62 = __adj53 * __adj41;
    const double __adj60 = -1.000000;
    const double __adj58 = __adj53 * __adj4;
    const double __adj61 = __adj60 * __adj58;
    const double result_d_r = __adj62 + __adj61;
    const double __adj63 = __symbol_11;
    const double result_d_S = __adj63 * __adj36;
    const double __adj66 = __statement_7;
    const double __adj67 = __adj66 * __adj21;
    const double __adj64 = d___symbol_8;
    const double __adj65 = __adj64 * __adj31;
    const double result_d_K = __adj67 + __adj65;
    const double __adj68 = __symbol_3;
    const double result_d_vol = __adj68 * __adj49;
    return std::array<double, 7>{result, result_d_t, result_d_T, result_d_r, result_d_S, result_d_K, result_d_vol};
}

template<class Kernel>
void PrintCode()
{
    auto ad_kernel = typename Kernel::template Build<DoubleKernel>();

    auto arguments = ad_kernel.Arguments();

    std::vector<DoubleKernel> symbolc_arguments;
    for (auto const& arg : arguments)
    {
        symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
    }
    auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

    auto head = function_root.as_operator_();

    auto three_address_transform = std::make_shared<Transform::RemapUnique>();
    auto three_address_tree = head->Clone(three_address_transform);

    //three_address_transform->Debug();

    //auto call_expanded_head = ExpandCall(three_address_tree);
    auto call_expanded_head = three_address_tree;

    auto block = BuildRecursiveEx(call_expanded_head);

    auto f = std::make_shared<Function>(block);
    for (auto const& arg : arguments)
    {
        f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
    }

    //f->GetModule()->EmitCode(std::cout);

    auto M = f->GetModule();

    auto v = std::make_shared< DebugControlBlockVisitor>();
    //M->Accept(*v);

    auto l = std::make_shared< InstructionLinearizer>();
    M->Accept(*l);

    auto ff = std::make_shared<ProgramCode::Function>(ad_kernel.Name(), arguments, l->stmts_);
    //ff->DebugPrint();

    //ProgramCode::CodeWriter{}.EmitCode(std::cout, ff);

    auto g = CloneWithDiffs(ff);

    ProgramCode::CodeWriter{}.EmitCode(std::cout, g);
}

auto __FactorPlus(const double sigma, const double r, const double tau, const double s)
{
    const double __symbol_3 = sigma;
    const double __symbol_15 = 1.000000 / __symbol_3;
    const double __symbol_1 = tau;
    const double __symbol_13 = std::pow(__symbol_1, 0.500000);
    const double __symbol_16 = __symbol_15 / __symbol_13;
    const double __symbol_9 = s;
    const double __symbol_10 = std::log(__symbol_9);
    const double __symbol_6 = r;
    const double __symbol_4 = std::pow(__symbol_3, 2.000000);
    const double __symbol_5 = __symbol_4 / 2.000000;
    const double __symbol_7 = __symbol_6 + __symbol_5;
    const double __symbol_8 = __symbol_7 * __symbol_1;
    const double __symbol_11 = __symbol_10 + __symbol_8;
    const double __symbol_17 = __symbol_16 * __symbol_11;
    const double __statement_0 = __symbol_17;
    const double result = __statement_0;
    const double __adj7 = __symbol_3;
    const double __adj18 = 2.000000 * __adj7;
    const double __adj16 = 2.000000 / 4.000000;
    const double __adj13 = __symbol_1;
    const double __adj12 = __symbol_16;
    const double __adj14 = __adj13 * __adj12;
    const double __adj17 = __adj16 * __adj14;
    const double __adj19 = __adj18 * __adj17;
    const double __adj8 = std::pow(__adj7, 2.000000);
    const double __adj10 = -1.000000 / __adj8;
    const double __adj3 = __symbol_13;
    const double __adj4 = std::pow(__adj3, 2.000000);
    const double __adj5 = __adj3 / __adj4;
    const double __adj1 = __symbol_11;
    const double __adj6 = __adj5 * __adj1;
    const double __adj11 = __adj10 * __adj6;
    const double result_d_sigma = __adj19 + __adj11;
    const double result_d_r = __adj13 * __adj12;
    const double __adj29 = __symbol_7;
    const double __adj30 = __adj29 * __adj12;
    const double __adj25 = std::pow(__adj13, -0.500000);
    const double __adj27 = 0.500000 * __adj25;
    const double __adj20 = __symbol_15;
    const double __adj21 = -__adj20;
    const double __adj22 = __adj21 / __adj4;
    const double __adj23 = __adj22 * __adj1;
    const double __adj28 = __adj27 * __adj23;
    const double result_d_tau = __adj30 + __adj28;
    const double __adj31 = __symbol_9;
    const double __adj33 = 1.000000 / __adj31;
    const double result_d_s = __adj33 * __adj12;
    return std::array<double, 5>{result, result_d_sigma, result_d_r, result_d_tau, result_d_s};
}
auto __FactorMinus(const double sigma, const double r, const double tau, const double s)
{
    const double __symbol_3 = sigma;
    const double __symbol_15 = 1.000000 / __symbol_3;
    const double __symbol_1 = tau;
    const double __symbol_13 = std::pow(__symbol_1, 0.500000);
    const double __symbol_16 = __symbol_15 / __symbol_13;
    const double __symbol_9 = s;
    const double __symbol_10 = std::log(__symbol_9);
    const double __symbol_6 = r;
    const double __symbol_4 = std::pow(__symbol_3, 2.000000);
    const double __symbol_5 = __symbol_4 / 2.000000;
    const double __symbol_7 = __symbol_6 - __symbol_5;
    const double __symbol_8 = __symbol_7 * __symbol_1;
    const double __symbol_11 = __symbol_10 + __symbol_8;
    const double __symbol_17 = __symbol_16 * __symbol_11;
    const double __statement_1 = __symbol_17;
    const double result = __statement_1;
    const double __adj7 = __symbol_3;
    const double __adj19 = 2.000000 * __adj7;
    const double __adj17 = 2.000000 / 4.000000;
    const double __adj13 = __symbol_1;
    const double __adj12 = __symbol_16;
    const double __adj14 = __adj13 * __adj12;
    const double __adj15 = -1.000000 * __adj14;
    const double __adj18 = __adj17 * __adj15;
    const double __adj20 = __adj19 * __adj18;
    const double __adj8 = std::pow(__adj7, 2.000000);
    const double __adj10 = -1.000000 / __adj8;
    const double __adj3 = __symbol_13;
    const double __adj4 = std::pow(__adj3, 2.000000);
    const double __adj5 = __adj3 / __adj4;
    const double __adj1 = __symbol_11;
    const double __adj6 = __adj5 * __adj1;
    const double __adj11 = __adj10 * __adj6;
    const double result_d_sigma = __adj20 + __adj11;
    const double result_d_r = __adj13 * __adj12;
    const double __adj30 = __symbol_7;
    const double __adj31 = __adj30 * __adj12;
    const double __adj26 = std::pow(__adj13, -0.500000);
    const double __adj28 = 0.500000 * __adj26;
    const double __adj21 = __symbol_15;
    const double __adj22 = -__adj21;
    const double __adj23 = __adj22 / __adj4;
    const double __adj24 = __adj23 * __adj1;
    const double __adj29 = __adj28 * __adj24;
    const double result_d_tau = __adj31 + __adj29;
    const double __adj32 = __symbol_9;
    const double __adj34 = 1.000000 / __adj32;
    const double result_d_s = __adj34 * __adj12;
    return std::array<double, 5>{result, result_d_sigma, result_d_r, result_d_tau, result_d_s};
}
auto __KoBarrierCallOption(const double x, const double K, const double tau, const double r, const double sigma, const double B)
{
    const double __symbol_5 = B;
    const double __symbol_23 = -__symbol_5;
    const double __symbol_4 = x;
    const double __symbol_21 = __symbol_4 / __symbol_5;
    const double __symbol_2 = r;
    const double __symbol_19 = -2.000000 * __symbol_2;
    const double __symbol_1 = sigma;
    const double __symbol_17 = std::pow(__symbol_1, 2.000000);
    const double __symbol_20 = __symbol_19 / __symbol_17;
    const double __symbol_22 = std::pow(__symbol_21, __symbol_20);
    const double __symbol_24 = __symbol_23 * __symbol_22;
    const double __symbol_3 = tau;
    const double __symbol_9 = K;
    const double __symbol_11 = 2.000000 / __symbol_9;
    const double __symbol_12 = __symbol_11 / __symbol_4;
    const double __symbol_13 = std::pow(__symbol_5, __symbol_12);
    auto __call_result_19020524 = __FactorPlus(__symbol_1, __symbol_2, __symbol_3, __symbol_13);
    const double __symbol_14 = __call_result_19020524[0];
    const double d_18929508___symbol_1 = __call_result_19020524[1];
    const double d_18929508___symbol_2 = __call_result_19020524[2];
    const double d_18929508___symbol_3 = __call_result_19020524[3];
    const double d_18929508___symbol_13 = __call_result_19020524[4];
    const double __symbol_15 = std::erfc(-(__symbol_14) / std::sqrt(2)) / 2;
    const double __symbol_6 = __symbol_5 / __symbol_4;
    auto __call_result_19020644 = __FactorMinus(__symbol_1, __symbol_2, __symbol_3, __symbol_6);
    const double __symbol_7 = __call_result_19020644[0];
    const double d_18929028___symbol_1 = __call_result_19020644[1];
    const double d_18929028___symbol_2 = __call_result_19020644[2];
    const double d_18929028___symbol_3 = __call_result_19020644[3];
    const double d_18929028___symbol_6 = __call_result_19020644[4];
    const double __symbol_8 = std::erfc(-(__symbol_7) / std::sqrt(2)) / 2;
    const double __symbol_16 = __symbol_15 - __symbol_8;
    const double __symbol_25 = __symbol_24 * __symbol_16;
    const double __statement_4 = __symbol_25;
    const double __statement_7 = __statement_4;
    const double result = __statement_7;
    const double __adj12 = __symbol_5;
    const double __adj32 = -__adj12;
    const double __adj30 = __symbol_4;
    const double __adj31 = std::pow(__adj30, 2.000000);
    const double __adj33 = __adj32 / __adj31;
    const double __adj28 = d_18929028___symbol_6;
    const double __adj20 = __symbol_7;
    const double __adj21 = std::pow(__adj20, 2.000000);
    const double __adj23 = 0.500000 * __adj21;
    const double __adj24 = -__adj23;
    const double __adj25 = std::exp(__adj24);
    const double __adj26 = __adj25 / 2.506628;
    const double __adj16 = __symbol_24;
    const double __adj18 = -1.000000 * __adj16;
    const double __adj27 = __adj26 * __adj18;
    const double __adj29 = __adj28 * __adj27;
    const double __adj34 = __adj33 * __adj29;
    const double __adj13 = std::pow(__adj12, 2.000000);
    const double __adj14 = __adj12 / __adj13;
    const double __adj5 = __symbol_20;
    const double __adj7 = __symbol_21;
    const double __adj6 = __adj5 - 1.000000;
    const double __adj8 = std::pow(__adj7, __adj6);
    const double __adj9 = __adj5 * __adj8;
    const double __adj2 = __symbol_23;
    const double __adj1 = __symbol_16;
    const double __adj3 = __adj2 * __adj1;
    const double __adj10 = __adj9 * __adj3;
    const double __adj15 = __adj14 * __adj10;
    const double result_d_x = __adj34 + __adj15;
    const double result_d_K = 0.000000;
    const double __adj44 = d_18929028___symbol_3;
    const double __adj45 = __adj44 * __adj27;
    const double __adj42 = d_18929508___symbol_3;
    const double __adj35 = __symbol_14;
    const double __adj36 = std::pow(__adj35, 2.000000);
    const double __adj37 = 0.500000 * __adj36;
    const double __adj38 = -__adj37;
    const double __adj39 = std::exp(__adj38);
    const double __adj40 = __adj39 / 2.506628;
    const double __adj41 = __adj40 * __adj16;
    const double __adj43 = __adj42 * __adj41;
    const double result_d_tau = __adj45 + __adj43;
    const double __adj48 = d_18929028___symbol_2;
    const double __adj49 = __adj48 * __adj27;
    const double __adj46 = d_18929508___symbol_2;
    const double __adj47 = __adj46 * __adj41;
    const double result_d_r = __adj49 + __adj47;
    const double __adj52 = d_18929028___symbol_1;
    const double __adj53 = __adj52 * __adj27;
    const double __adj50 = d_18929508___symbol_1;
    const double __adj51 = __adj50 * __adj41;
    const double result_d_sigma = __adj53 + __adj51;
    const double __adj68 = __adj30 / __adj31;
    const double __adj69 = __adj68 * __adj29;
    const double __adj63 = __symbol_12;
    const double __adj64 = __adj63 - 1.000000;
    const double __adj65 = std::pow(__adj12, __adj64);
    const double __adj66 = __adj63 * __adj65;
    const double __adj61 = d_18929508___symbol_13;
    const double __adj62 = __adj61 * __adj41;
    const double __adj67 = __adj66 * __adj62;
    const double __adj70 = __adj69 + __adj67;
    const double __adj58 = -__adj30;
    const double __adj59 = __adj58 / __adj13;
    const double __adj60 = __adj59 * __adj10;
    const double __adj71 = __adj70 + __adj60;
    const double __adj56 = -1.000000;
    const double __adj54 = __symbol_22;
    const double __adj55 = __adj54 * __adj1;
    const double __adj57 = __adj56 * __adj55;
    const double result_d_B = __adj71 + __adj57;
    return std::array<double, 7>{result, result_d_x, result_d_K, result_d_tau, result_d_r, result_d_sigma, result_d_B};
}


int main()
{
    PrintCode< KoBarrierOption::FactorPlus>();
    PrintCode< KoBarrierOption::FactorMinus>();
    PrintCode< KoBarrierOption::KoBarrierCallOption>();

    using kernel_ty = KoBarrierOption::KoBarrierCallOption;

    /*
    * Double x, 
                Double K,
                Double tau,
                Double r,
                Double sigma,
                Double B
    */
    double S = 80;
    double K = 100;
    double tau = 1.0;
    double r = 0.00;
    double vol = 0.2;
    double B = 120;
    std::cout << kernel_ty::Build<double>{}.Evaluate(S, K, tau, r, vol, B) << "\n";
    auto aad_result = __KoBarrierCallOption(S, K, tau, r, vol, B);



    std::vector<double> X{ S, K, tau, r, vol, B };
    std::cout << kernel_ty::Build<double>{}.EvaluateVec(X) << "\n";
    std::cout << "numeric=" << kernel_ty::Build<double>{}.EvaluateVec(X) << ", AAD=" << aad_result[0] << "\n";
    for (size_t idx = 0; idx != X.size(); ++idx)
    {
        const double epsilon = 1e-8;
        const double old_value = X[idx];
        X[idx] = old_value - epsilon / 2;
        auto left = kernel_ty::Build<double>{}.EvaluateVec(X);
        X[idx] = old_value + epsilon / 2;
        auto right = kernel_ty::Build<double>{}.EvaluateVec(X);

        auto numeric = (right - left) / epsilon;

        std::cout << "numeric=" << numeric << ", AAD=" << aad_result[idx + 1] << "\n";
    }
}
#if 0
int main()
{
    //test_bs();
    
    // using kernel_ty = BlackScholesCallOptionTest;
    //using kernel_ty = MyLogKFDivStd;

    //using kernel_ty = KoBarrierOption::KoBarrierCallOption;
    using kernel_ty = KoBarrierOption::FactorMinus;

#if 0
    double t = 0.0;
    double T = 2.0;
    double r = 0.00;
    double S = 80;
    double K = 100;
    double vol = 0.2;
    double B = 120;
    std::cout << kernel_ty::Build<double>{}.Evaluate(t, T, r, S, K, vol, B) << "\n";
    auto aad_result = __black(t, T, r, S, K, vol, );



    std::vector<double> X{ t, T, r, S, K, vol };
    std::cout << kernel_ty::Build<double>{}.EvaluateVec(X) << "\n";
    std::cout << "numeric=" << kernel_ty::Build<double>{}.EvaluateVec(X) << ", AAD=" << aad_result[0] << "\n";
    for (size_t idx = 0; idx != X.size(); ++idx)
    {
        const double epsilon = 1e-8;
        const double old_value = X[idx];
        X[idx] = old_value - epsilon/2;
        auto left = kernel_ty::Build<double>{}.EvaluateVec(X);
        X[idx] = old_value + epsilon / 2;
        auto right = kernel_ty::Build<double>{}.EvaluateVec(X);

        auto numeric = (right - left) / epsilon;

        std::cout << "numeric=" << numeric << ", AAD=" << aad_result[idx+1] << "\n";
    }
#endif


    auto ad_kernel = kernel_ty::Build<DoubleKernel>();

    auto arguments = ad_kernel.Arguments();

    std::vector<DoubleKernel> symbolc_arguments;
    for (auto const& arg : arguments)
    {
        symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
    }
    auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

    auto head = function_root.as_operator_();

    auto three_address_transform = std::make_shared<Transform::RemapUnique>();
    auto three_address_tree = head->Clone(three_address_transform);

    //three_address_transform->Debug();

    //auto call_expanded_head = ExpandCall(three_address_tree);
    auto call_expanded_head = three_address_tree;

    auto block = BuildRecursiveEx(call_expanded_head);

    auto f = std::make_shared<Function>(block);
    for (auto const& arg : arguments)
    {
        f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
    }

    //f->GetModule()->EmitCode(std::cout);

    auto M = f->GetModule();

    auto v = std::make_shared< DebugControlBlockVisitor>();
    //M->Accept(*v);

    auto l = std::make_shared< InstructionLinearizer>();
    M->Accept(*l);

    auto ff = std::make_shared<ProgramCode::Function>(ad_kernel.Name(), arguments, l->stmts_);
    //ff->DebugPrint();

    //ProgramCode::CodeWriter{}.EmitCode(std::cout, ff);

    auto g = CloneWithDiffs(ff);

    ProgramCode::CodeWriter{}.EmitCode(std::cout, g);
}
#endif

