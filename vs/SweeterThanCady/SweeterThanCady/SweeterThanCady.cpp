// SweeterThanCady.cpp : Defines the entry point for the application.
//

#include "SweeterThanCady.h"


#include "Cady/CodeGen.h"
#include "Cady/Frontend.h"
#include "Cady/Transform.h"
#include "Cady/Cady.h"

#include <map>

using namespace Cady;




enum InstructionKind {
    Instr_VarDecl,
    Instr_PointerAssignment,
    Instr_Return,
    Instr_Comment,
};
struct Instruction {
    explicit Instruction(InstructionKind kind)
        :kind_(kind)
    {}
    virtual ~Instruction() = default;
    InstructionKind Kind()const { return kind_; }
    virtual void EmitCode(std::ostream& out)const = 0;
private:
    InstructionKind kind_;
};
struct InstructionComment : Instruction {
    InstructionComment(std::vector<std::string> const& text)
        :Instruction{ Instr_Comment }
        , text_{ text }
    {}
    virtual void EmitCode(std::ostream& out)const {
        for (auto const& comment : text_) {
            out << "    // " << comment << "\n";
        }
    }
private:
    std::vector<std::string> text_;
};
struct InstructionDeclareVariable : Instruction {
    InstructionDeclareVariable(std::string const& name, std::shared_ptr<Operator> op)
        :Instruction{ Instr_VarDecl }
        , name_(name)
        , op_(op)
    {}
    virtual void EmitCode(std::ostream& out)const {
        out << "    double const " << name_ << " = ";
        op_->EmitCode(out);
        out << ";\n";
    }
    auto const& as_operator_()const { return op_; }
private:
    std::string name_;
    std::shared_ptr<Operator> op_;
};
struct InstructionPointerAssignment : Instruction {
    InstructionPointerAssignment(std::string const& name, std::string const& r_value)
        :Instruction{ Instr_PointerAssignment }
        , name_(name)
        , r_value_{ r_value }
    {}
    virtual void EmitCode(std::ostream& out)const {
        out << "    *" << name_ << " = " << r_value_ << ";\n";
    }
private:
    std::string name_;
    std::string r_value_;
};
struct InstructionReturn : Instruction {
    InstructionReturn(std::string const& name)
        :Instruction{ Instr_Return }
        , name_(name)
    {}
    virtual void EmitCode(std::ostream& out)const {
        out << "    return " << name_ << ";\n";
    }
private:
    std::string name_;
};

struct InstructionBlock : std::vector<std::shared_ptr<Instruction> > {
    virtual ~InstructionBlock() = default;
    void Add(std::shared_ptr<Instruction> instr) {
        // quick hack for now
        if (size() > 0 && back()->Kind() == Instr_Return) {
            auto tmp = back();
            pop_back();
            push_back(instr);
            push_back(tmp);
        }
        else {
            push_back(instr);
        }
    }
    virtual void EmitCode(std::ostream& out)const {
        for (auto const& ptr : *this) {
            ptr->EmitCode(out);
        }
    }
};



struct BlackScholesCallOption {
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
#if 0
            using MathFunctions::Phi;
            using MathFunctions::Exp;
            using MathFunctions::Pow;
            using MathFunctions::Log;

            Double d1 = ((1.0 / (vol * Pow((T - t), 0.5))) * (Log(S / K) + (r + (Pow(vol, 2.0)) / 2) * (T - t)));
            Double d2 = d1 - vol * (T - t);
            Double pv = K * Exp(-r * (T - t));
            Double black = Phi(d1) * S - Phi(d2) * pv;
            return black;
#endif
            Double a0 = t * t;
            Double a1 = T * T;
            Double a2 = r * r;
            Double a3 = S * S;
            Double a4 = K * K;
            Double a5 = vol * vol;

            Double b0 = a0 * a1 + a2;
            Double b1 = a1 * a2 + a3;
            Double b2 = a2 * a3 + a4;
            Double b3 = a3 * a4 + a5;
            Double b4 = a4 * a5 + a0;

            Double c1 = b0 * b1 + b3;
            Double c2 = b1 * b2 + b3;

            Double result = c2 * c1;

            return result;
            
        }
    };
};



struct ProfileFunctionDefinition {
    std::string BaseName;
    std::string FunctionName;
    std::vector<std::string> Args;
    std::vector<std::string> Fields;
};

struct ProfileFunction {
    std::string ImplName;
    std::shared_ptr<InstructionBlock> IB;
    std::unordered_map<std::string, std::string> ResultMapping;
};

struct ProfileCodeGen {
    explicit ProfileCodeGen(std::shared_ptr<ProfileFunctionDefinition> const& def) :def_{ def } {}
    void AddImplementation(std::shared_ptr<ProfileFunction> func) {
        vec_.push_back(func);
    }
    void EmitCode(std::ostream& out = std::cout) {
        out << "#include <vector>\n";
        out << "#include <unordered_map>\n";
        out << "#include <cstdio>\n";
        out << "#include <cmath>\n";
        out << "#include <iostream>\n";
        out << "#include <memory>\n";
        out << "#include <chrono>\n";
        out << "#include <string>\n";
        out << "\n";
        out << "struct cpu_timer{\n";
        out << "    cpu_timer()\n";
        out << "        : start_{std::chrono::high_resolution_clock::now()}\n";
        out << "    {}\n";
        out << "    template<class... Args>\n";
        out << "    std::string format()const {\n";
        out << "        return std::to_string((std::chrono::high_resolution_clock::now() - start_).count());\n";
        out << "    }\n";
        out << "    std::chrono::time_point<std::chrono::high_resolution_clock> start_;\n";
        out << "};\n";
        out << "\n";
        out << "struct " << def_->BaseName << "{\n";
        out << "    struct ResultType{\n";
        for (auto const& field : def_->Fields) {
            out << "        double " << field << " = 0.0;\n";
        };
        out << "    };\n";
        out << "    virtual ~" << def_->BaseName << "()=default;\n";
        out << "    virtual ResultType " << def_->FunctionName << "(\n";
        for (size_t idx = 0; idx != def_->Args.size(); ++idx) {
            out << "        " << (idx == 0 ? "  " : ", ") << "double " << def_->Args[idx] << (idx + 1 == def_->Args.size() ? ")const=0;\n" : "\n");
        }
        out << "    using map_ty = std::unordered_map<std::string, std::shared_ptr<" << def_->BaseName << "> >;\n";
        out << "    static map_ty& AllImplementations(){\n";
        out << "            static map_ty mem;\n";
        out << "            return mem;\n";
        out << "    }\n";
        out << "};\n";
        out << "\n";
        auto RegisterBaseBame = "Register" + def_->BaseName;
        out << "template<class T>\n";
        out << "struct " << RegisterBaseBame << "{\n";
        out << "    template<class... Args>\n";
        out << "    " << RegisterBaseBame << "(std::string const& name, Args&&... args){\n";
        out << "        " << def_->BaseName << "::AllImplementations()[name] = std::make_shared<T>(std::forward<Args>(args)...);\n";
        out << "    }\n";
        out << "};\n";
        out << "\n";
        


        for (auto func : vec_) {
            out << "struct " << func->ImplName << " : " << def_->BaseName << "{\n";
            out << "    virtual ResultType " << def_->FunctionName << "(\n";
            for (size_t idx = 0; idx != def_->Args.size(); ++idx) {
                out << "        " << (idx == 0 ? "  " : ", ") << "double " << def_->Args[idx] << (idx + 1 == def_->Args.size() ? ")const override{\n" : "\n");
            }
            func->IB->EmitCode(out);
            out << "        ResultType result;\n";
            for (auto const& p : func->ResultMapping) {
                out << "        result." << p.first << " = " << p.second << ";\n";
            }
            out << "        return result;\n";
            out << "    }\n";
            out << "};\n";
            out << "static " << RegisterBaseBame << "<" << func->ImplName << "> Reg" << func->ImplName << "(\"" << func->ImplName << "\");\n";
        }


        out << "int main(){\n";
        out << "    enum{ InitialCount = 10000 };\n";
        out << "    double t       = 0.0;\n";
        out << "    double T       = 10.0;\n";
        out << "    double r       = 0.04;\n";
        out << "    double S       = 50;\n";
        out << "    double K       = 60;\n";
        out << "    double vol     = 0.2;\n";
        out << "    double epsilon = 1e-10;\n";
        out << "\n";
        out << "    // get all implementations\n";
        out << "    auto impls = " << def_->BaseName << "::AllImplementations();\n";
        out << "\n";
        out << "    // print header\n";
        out << "    std::vector<std::string> header{\"N\"}; \n";
        out << "    for(auto impl : impls ){\n";
        out << "        header.push_back(impl.first);\n";
        out << "    }\n";
        out << "    auto print_line = [](std::vector<std::string> const& line_vec){ for(size_t idx=0;idx!=line_vec.size();++idx){ std::cout << (idx==0?\"\":\",\") << line_vec[idx];} std::cout << '\\n'; };\n";
        out << "    print_line(header);\n";
        out << "\n";
        out << "\n";
        enum{ PrintCheck = 1 };
        if (PrintCheck)
        {
            out << "    std::vector<std::string> check;\n";
            out << "    auto proto_name = \"NaiveBlack\";\n";
            out << "    auto proto = impls[proto_name];\n";
            out << "    auto baseline = proto->" << def_->FunctionName << "(t,T,r,S,K,vol);\n";
            for (auto const& field : def_->Fields) {
                out << "    check.clear();\n";
                out << "    check.push_back(\"" << field << "\");\n";
                out << "    for(auto impl : impls ){\n";
                out << "        check.push_back(std::to_string(impl.second->" << def_->FunctionName << "(t,T,r,S,K,vol)." << field << "));\n";
                out << "    }\n";
                out << "    check.push_back(std::to_string((proto->" << def_->FunctionName << "(";
                for (size_t idx = 0; idx != def_->Args.size(); ++idx) {
                    out << (idx == 0 ? "" : ",") << def_->Args[idx];
                    if ("d_" + def_->Args[idx] == field) {
                        out << "+epsilon";
                    }
                }
                out << ").c-baseline.c)/epsilon));\n";

                out << "    print_line(check);\n";
            }
        }
       
        out << "    for(volatile size_t N = 100;N<100000;N*=2){\n";
        out << "        std::vector<std::string> line;\n";
        out << "        line.push_back(std::to_string(N));\n";
        out << "        for(auto impl : impls ){\n";
        out << "            cpu_timer timer;\n";
        out << "            for(volatile size_t idx=0;idx!=N;++idx){\n";
        out << "                auto result = impl.second->" << def_->FunctionName << "(t,T,r,S,K,vol);\n";
        out << "            }\n";
        out << "            line.push_back(timer.format());\n";
        out << "        }\n";
        out << "        print_line(line);\n";
        out << "    }\n";
        out << "}\n";


    }
private:
    std::shared_ptr<ProfileFunctionDefinition> def_;
    std::vector<std::shared_ptr<ProfileFunction> > vec_;
};


struct NaiveBlackProfile : ProfileFunction {
    NaiveBlackProfile() {
        ImplName = "NaiveBlack";
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate(
            DoubleKernel::BuildFromExo("t"),
            DoubleKernel::BuildFromExo("T"),
            DoubleKernel::BuildFromExo("r"),
            DoubleKernel::BuildFromExo("S"),
            DoubleKernel::BuildFromExo("K"),
            DoubleKernel::BuildFromExo("vol")
        );

        auto expr = as_black.as_operator_();

        IB = std::make_shared<InstructionBlock>();
        auto deps = expr->DepthFirstAnySymbolicDependencyAndThis();
        for (auto head : deps.DepthFirst) {
            if (head->IsExo())
                continue;
            auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
            IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
        }
        ResultMapping["c"] = deps.DepthFirst.back()->Name();
    }
};

struct NaiveBlackSingleProfile : ProfileFunction {
    NaiveBlackSingleProfile() {
        ImplName = "NaiveBlackSingle";
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate(
            DoubleKernel::BuildFromExo("t"),
            DoubleKernel::BuildFromExo("T"),
            DoubleKernel::BuildFromExo("r"),
            DoubleKernel::BuildFromExo("S"),
            DoubleKernel::BuildFromExo("K"),
            DoubleKernel::BuildFromExo("vol")
        );

        auto expr = as_black.as_operator_();

        struct RemoveEndo : OperatorTransform {
            virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                auto candidate = ptr->Clone(shared_from_this());
                if (candidate->Kind() == OPKind_EndgenousSymbol) {
                    if (auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)) {
                        return typed->Expr();
                    }
                }
                return candidate;
            }
        };
        auto single_expr = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemoveEndo>());


        IB = std::make_shared<InstructionBlock>();
        IB->Add(std::make_shared<InstructionDeclareVariable>("c", single_expr));
        ResultMapping["c"] = "c";

    }
};
struct RemapUnique : OperatorTransform {
    explicit RemapUnique(std::string const& prefix = "__symbol_")
        : prefix_{ prefix }
    {
        std::cerr << " RemapUnique()\n";
    }
    ~RemapUnique() {
        std::cerr << "~RemapUnique()\n";
    }
    void mutate_prefix(std::string const& prefix) {
        prefix_ = prefix;
    }
    virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {

        auto candidate = ptr->Clone(shared_from_this());

        auto key = std::make_tuple(
            candidate->NameInvariantOfChildren(),
            candidate->Children()
        );

        auto iter = ops_.find(key);
        if (iter != ops_.end())
            return iter->second;

        if (
            candidate->Kind() != OPKind_EndgenousSymbol &&
#if 0
            candidate->Kind() != OPKind_ExogenousSymbol &&
#endif
            candidate->Kind() != OPKind_Constant)
        {

            std::stringstream ss;
            ss << prefix_ << (ops_.size() + 1);
            auto endogous_sym = EndgenousSymbol::Make(ss.str(), candidate);

            ops_[key] = endogous_sym;
            return endogous_sym;
        }
        else {
            ops_[key] = candidate;
            return candidate;
        }
    }
private:
    std::string prefix_;
    std::map<
        std::tuple<
        std::string,
        std::vector<std::shared_ptr<Operator> >
        >,
        std::shared_ptr<Operator>
    > ops_;
};
struct NaiveBlackThreeAddressProfile : ProfileFunction {
    NaiveBlackThreeAddressProfile() {
        ImplName = "NaiveBlackThreeAddress";
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate(
            DoubleKernel::BuildFromExo("t"),
            DoubleKernel::BuildFromExo("T"),
            DoubleKernel::BuildFromExo("r"),
            DoubleKernel::BuildFromExo("S"),
            DoubleKernel::BuildFromExo("K"),
            DoubleKernel::BuildFromExo("vol")
        );

        auto expr = as_black.as_operator_();
        //auto unique = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemapUnique>());
        auto unique = expr->Clone(std::make_shared<RemapUnique>());



        IB = std::make_shared<InstructionBlock>();
        auto deps = unique->DepthFirstAnySymbolicDependencyAndThis();
        for (auto head : deps.DepthFirst) {
            if (head->IsExo())
                continue;
            auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
            IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
        }
        ResultMapping["c"] = deps.DepthFirst.back()->Name();

    }
};

struct NaiveBlackForwardThreeAddressProfile : ProfileFunction {
    NaiveBlackForwardThreeAddressProfile() {
        ImplName = "NaiveBlackForwardThreeAddress";
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate(
            DoubleKernel::BuildFromExo("t"),
            DoubleKernel::BuildFromExo("T"),
            DoubleKernel::BuildFromExo("r"),
            DoubleKernel::BuildFromExo("S"),
            DoubleKernel::BuildFromExo("K"),
            DoubleKernel::BuildFromExo("vol")
        );

        auto expr = as_black.as_operator_();
        //auto unique = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemapUnique>());
        auto RU = std::make_shared<RemapUnique>();
        auto unique = expr->Clone(RU);


        std::unordered_set<std::string> three_addr_seen;

        IB = std::make_shared<InstructionBlock>();
        auto deps = unique->DepthFirstAnySymbolicDependencyAndThis();
        for (auto head : deps.DepthFirst) {
            if (head->IsExo())
                continue;
            if (three_addr_seen.count(head->Name()) == 0) {
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                three_addr_seen.insert(head->Name());
            }
        }
        ResultMapping["c"] = deps.DepthFirst.back()->Name();

        struct RemoveEndo : OperatorTransform {
            virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                auto candidate = ptr->Clone(shared_from_this());
                if (candidate->Kind() == OPKind_EndgenousSymbol) {
                    if (auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)) {
                        return typed->Expr();
                    }
                }
                return candidate;
            }
        };
        auto single_expr = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemoveEndo>());

        std::vector<std::string> exo{ "t", "T", "r", "S", "K", "vol" };
#if 1
        for (auto sym : exo) {
            auto sym_diff = single_expr->Diff(sym);
#if 0
            auto sym_diff_unique = EndgenousSymbol::Make("d_" + sym, sym_diff->Clone(std::make_shared<RemapUnique>("__d_" + sym)));
#endif
            auto sym_diff_unique = EndgenousSymbol::Make("d_" + sym, sym_diff->Clone(RU));
            auto deps = sym_diff_unique->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) == 0) {
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                    IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                    three_addr_seen.insert(head->Name());
                }
            }
            ResultMapping["d_" + sym] = deps.DepthFirst.back()->Name();
        }
#else
        for (auto sym : exo) {
            IB->Add(std::make_shared<InstructionDeclareVariable>("d_" + sym, single_expr->Diff(sym)));
            ResultMapping["d_" + sym] = "d_" + sym;
        }
#endif
    }
};

struct NaiveBlackADThreeAddressProfile : ProfileFunction {
    NaiveBlackADThreeAddressProfile() {
        ImplName = "NaiveBlackADThreeAddress";
        auto ad_kernel = BlackScholesCallOption::Build<DoubleKernel>{};

        auto as_black = ad_kernel.Evaluate(
            DoubleKernel::BuildFromExo("t"),
            DoubleKernel::BuildFromExo("T"),
            DoubleKernel::BuildFromExo("r"),
            DoubleKernel::BuildFromExo("S"),
            DoubleKernel::BuildFromExo("K"),
            DoubleKernel::BuildFromExo("vol")
        );

        auto expr = as_black.as_operator_();
        //auto unique = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemapUnique>());
        auto RU = std::make_shared<RemapUnique>();
        auto unique = expr->Clone(RU);


        std::unordered_set<std::string> three_addr_seen;

        IB = std::make_shared<InstructionBlock>();
        auto deps = unique->DepthFirstAnySymbolicDependencyAndThis();
        for (auto head : deps.DepthFirst) {
            if (head->IsExo())
                continue;
            if (three_addr_seen.count(head->Name()) == 0) {
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                three_addr_seen.insert(head->Name());
            }
        }
        ResultMapping["c"] = deps.DepthFirst.back()->Name();

        struct RemoveEndo : OperatorTransform {
            virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                auto candidate = ptr->Clone(shared_from_this());
                if (candidate->Kind() == OPKind_EndgenousSymbol) {
                    if (auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)) {
                        return typed->Expr();
                    }
                }
                return candidate;
            }
        };
        auto single_expr = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemoveEndo>());
        auto expr_deps = unique->DepthFirstAnySymbolicDependencyAndThis();

        std::unordered_map<std::string, std::shared_ptr<Operator> > adj_expr;
        for (auto head : expr_deps.DepthFirst) {
            adj_expr[head->Name()] = Constant::Make(0.0);
        }
        adj_expr[expr_deps.DepthFirst.back()->Name()] = Constant::Make(1.0);

        for (size_t idx = expr_deps.DepthFirst.size(); idx != 0;) {
            --idx;
            auto head = expr_deps.DepthFirst[idx];
            if (head->IsExo())
                continue;
            auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
            auto subs = head->DepthFirstAnySymbolicDependencyNoRecurse();
            for (auto ptr : subs.DepthFirst) {
                adj_expr[ptr->Name()] = BinaryOperator::Add(
                    adj_expr[ptr->Name()],
                    BinaryOperator::Mul(
                        adj_expr[head->Name()],
                        expr->Diff(ptr->Name())
                    )
                );
            }
        }
        for (auto p : adj_expr) {
            std::cout << p.first << " => ";
            p.second->EmitCode(std::cout);
            std::cout << "\n";
        }
        std::vector<std::string> exo{ "t", "T", "r", "S", "K", "vol" };
        for (auto sym : exo) {
            auto sym_diff_unique = EndgenousSymbol::Make("d_" + sym, adj_expr[sym]->Clone(RU));
            auto deps = sym_diff_unique->DepthFirstAnySymbolicDependencyAndThis();
            for (auto head : deps.DepthFirst) {
                if (head->IsExo())
                    continue;
                if (three_addr_seen.count(head->Name()) == 0) {
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                    IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                    three_addr_seen.insert(head->Name());
                }
            }
            ResultMapping["d_" + sym] = deps.DepthFirst.back()->Name();
        }
    }
};

int main()
{
    auto def = std::make_shared<ProfileFunctionDefinition>();
    def->BaseName = "BlackPricer";
    def->FunctionName = "Black";
    def->Args.push_back("t");
    def->Args.push_back("T");
    def->Args.push_back("r");
    def->Args.push_back("S");
    def->Args.push_back("K");
    def->Args.push_back("vol");
    def->Fields.push_back("c");
    def->Fields.push_back("d_t");
    def->Fields.push_back("d_T");
    def->Fields.push_back("d_r");
    def->Fields.push_back("d_S");
    def->Fields.push_back("d_K");
    def->Fields.push_back("d_vol");
    ProfileCodeGen cg(def);



    cg.AddImplementation(std::make_shared<NaiveBlackProfile>());
    cg.AddImplementation(std::make_shared<NaiveBlackSingleProfile>());
    cg.AddImplementation(std::make_shared<NaiveBlackThreeAddressProfile>());
    cg.AddImplementation(std::make_shared<NaiveBlackForwardThreeAddressProfile>());
    cg.AddImplementation(std::make_shared<NaiveBlackADThreeAddressProfile>());


    // cg.EmitCode();

    std::ofstream out("pc.cxx");
    cg.EmitCode(out);
}
