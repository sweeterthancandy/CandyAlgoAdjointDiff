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

            Double special_condition = t * T - r * S;

            return If(
                special_condition,
                [&]() {
                    //Double tmp = Max(S - K, special_condition);

                    Double tmp = Call(MyMax{}, S - K, special_condition);
                    return tmp;
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
                auto name = "__tmp_call_" + std::to_string(names_.size());
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
        };

        struct DoubleConstant : public RValue
        {
            explicit DoubleConstant(double value) :value_{ value } {}
            double Value()const { return value_; }
            virtual std::string ToString()const override { return std::to_string(value_); }
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

#if 0
        class CallStatement : public Statement
        {
        private:
            std::shared_ptr<FunctionDecl> function_decl_;
            std::vector<std::string> result_list_;
            std::vector<std::shared_ptr<RValue> > arg_list_;
        };
#endif

        struct ReturnStatement : public Statement
        {
            explicit ReturnStatement(std::shared_ptr<RValue> const& value) :value_{ value } {}

            std::shared_ptr<RValue> value_;
        };


        struct Function
        {
            Function(std::vector<std::shared_ptr<Statement> > const& stmts)
                : stmts_{ stmts }
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
                else
                {
                    std::string();
                }
            }
            void DebugPrint()const
            {
                for (auto const& stmt : stmts_)
                {
                    DebugPrintStmt(stmt, 0);
                }
            }
        private:
            std::vector<std::shared_ptr<Statement> > stmts_;
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
            if (auto as_call = std::dynamic_pointer_cast<const Call>(as_lvalue_assign->as_operator_()))
            {
                std::cout << " // is a call site\n";
            }
            else
            {
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
    std::vector<std::shared_ptr<ProgramCode::Statement> > stmts_;
};







int main()
{
    //test_bs();
    
    using kernel_ty = BlackScholesCallOptionTest;
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

    auto call_expanded_head = ExpandCall(three_address_tree);


    auto block = BuildRecursiveEx(call_expanded_head);

    auto f = std::make_shared<Function>(block);
    for (auto const& arg : arguments)
    {
        f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
    }

    f->GetModule()->EmitCode(std::cout);

    auto M = f->GetModule();

    auto v = std::make_shared< DebugControlBlockVisitor>();
    M->Accept(*v);

    auto l = std::make_shared< InstructionLinearizer>();
    M->Accept(*l);

    auto ff = std::make_shared<ProgramCode::Function>(l->stmts_);
    ff->DebugPrint();
}