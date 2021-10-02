// Cady.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

// SweeterThanCady.cpp : Defines the entry point for the application.
//


#include "Cady/CodeGen.h"
#include "Cady/Frontend.h"
#include "Cady/Transform.h"
#include "Cady/Cady.h"

#include <map>

using namespace Cady;




enum InstructionKind {
    Instr_VarDecl,
    Instr_MatrixDecl,
    Instr_Text,
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
struct InstructionText : Instruction {
    InstructionText(std::string const& text)
        :Instruction{ Instr_Text }
        , text_{ text }
    {}
    virtual void EmitCode(std::ostream& out)const {
        out << "    " << text_ << "\n";
    }
private:
    std::string text_;
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
    std::string const& LValueName()const { return name_;  }
private:
    std::string name_;
    std::shared_ptr<Operator> op_;
};


struct InstructionDeclareMatrix : Instruction {
    InstructionDeclareMatrix(std::string const& name, std::vector<std::vector<std::shared_ptr<Operator> > > matrix)
        :Instruction{ Instr_VarDecl }
        , name_(name)
        , matrix_(matrix)
    {}
    virtual void EmitCode(std::ostream& out)const {
        size_t rows = matrix_.size();
        size_t cols = matrix_[0].size();
        out << "    const Eigen::Matrix<double, " << rows << "," << cols << ">" << name_ << " {\n";
        for (size_t i = 0; i != rows; ++i)
        {
            out << (i == 0 ? "" : ", ") << "{";
            for (size_t j = 0; j != cols; ++j)
            {
                out << (j == 0 ? "": ", ");
                matrix_.at(i).at(j)->EmitCode(out);
            }
            out << "}\n";
        }
        out << "};\n";
    }
    std::string const& LValueName()const { return name_; }
private:
    std::string name_;
    std::vector<std::vector<std::shared_ptr<Operator> > > matrix_;
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
        enum { PrintCheck = 1 };
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
    }
    ~RemapUnique() {
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

struct SimpleTest0 {
    template<class Double>
    struct Build {
        Double Evaluate(
            Double x,
            Double y)
            const
        {
            return x * x + y * y * y;

        }
    };
};

struct SimpleTest3 {
    template<class Double>
    struct Build {
        Double Evaluate(
            Double x,
            Double y,
            Double z)const
        {
            Double a0 = x * x;
            Double a1 = y * y;
            Double a2 = z * z;
            Double a3 = x * y;
            Double a4 = x * z;
            Double a5 = y * z;

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

struct ImpliedMatrixFunction
{
    ImpliedMatrixFunction(
        size_t id,
        std::string const& output,
        std::vector<std::shared_ptr<Symbol> > const& args,
        std::vector<std::shared_ptr<Operator> > const& diffs)
        : id_{ id }
        , output_{output}
        , args_ {args }
        , diffs_{ diffs }
    {}
    static std::shared_ptr< ImpliedMatrixFunction> Make(size_t id, std::shared_ptr<Instruction> const& instr)
    {
        if (auto decl_instr = std::dynamic_pointer_cast<InstructionDeclareVariable>(instr))
        {
            auto expr = decl_instr->as_operator_();
            auto deps = expr->DepthFirstAnySymbolicDependencyOrThisNoRecurse();
            std::vector<std::shared_ptr<Symbol> > input_sym(
                deps.Set.begin(), deps.Set.end());
            std::vector<std::shared_ptr<Operator> > diff_vec;
            for (auto const& sym : input_sym) {
                Transform::FoldZero fold_zero;

                auto diff = fold_zero.Fold(expr->Diff(sym->Name()));
                diff_vec.push_back(diff);
            }
            return std::make_shared< ImpliedMatrixFunction>(id, decl_instr->LValueName(), input_sym, diff_vec);
        }
        throw std::runtime_error("not implemented");
        
    }
    std::shared_ptr<InstructionComment> MakeComment()const
    {
        std::vector<std::string> comment_vec;

        comment_vec.push_back("Matrix function F_" + std::to_string(id_) + " => " + output_);
        for(size_t idx=0;idx!=args_.size();++idx)
        {
            std::stringstream ss;
            ss << "    " << args_[idx]->Name() << " => ";
            diffs_[idx]->EmitCode(ss);
            std::string comment = ss.str();
            if (comment.back() == '\n')
            {
                comment.pop_back();
            }
            comment_vec.push_back(comment);


        }

        return std::make_shared<InstructionComment>(comment_vec);
    }

    std::shared_ptr<InstructionDeclareMatrix> MakeMatrix(std::unordered_map<std::string,size_t> const& alloc, bool is_terminal)const
    {
        auto find_slot= [&](std::string const& name)
        {
            auto iter = alloc.find(name);
            if (iter == alloc.end())
            {
                throw std::runtime_error("bad alloctedd slot");
            }
            return iter->second;
        };

        auto zero = Constant::Make(0.0);
        auto one = Constant::Make(1.0);

        size_t n = alloc.size();
       
        std::vector<std::shared_ptr<Operator> > diff_col(n, zero);
        for (size_t idx = 0; idx != args_.size(); ++idx)
        {
            auto const& sym = args_[idx];
            auto const& diff = diffs_[idx];

            auto j = find_slot(sym->Name());

            diff_col[j] = diff;
        }

       


        std::vector<std::vector<std::shared_ptr<Operator> > > matrix(n);
        if (is_terminal)
        {
            for (size_t idx = 0; idx != n; ++idx)
            {
                matrix[idx].push_back(diff_col[idx]);
            }
        }
        else
        {
            for (auto& row : matrix)
            {
                row.resize(n + 1, zero);
            }
            // write identity matrix

            for (size_t i = 0; i != n; ++i)
            {
                matrix[i][i] = one;
            }

            for (size_t idx = 0; idx != n; ++idx)
            {
                matrix[idx][n] = diff_col[idx];
            }
        }
       

        std::string name = "adj_matrix_" + std::to_string(id_);
        return std::make_shared<InstructionDeclareMatrix>(name, matrix);

    }


private:
    size_t id_;
    std::string output_;
    std::vector<std::shared_ptr<Symbol> > args_;
    std::vector<std::shared_ptr<Operator> > diffs_;
};



int main__()
{
    auto ad_kernel = SimpleTest0::Build<DoubleKernel>{};

    auto as_black = ad_kernel.Evaluate(
        DoubleKernel::BuildFromExo("x"),
        DoubleKernel::BuildFromExo("y")
    );

    auto expr = as_black.as_operator_();
    //auto unique = std::reinterpret_pointer_cast<EndgenousSymbol>(expr)->Expr()->Clone(std::make_shared<RemapUnique>());
    auto RU = std::make_shared<RemapUnique>();
    auto unique = expr->Clone(RU);


    std::unordered_set<std::string> three_addr_seen;

    auto IB = std::make_shared<InstructionBlock>();
    auto AADIB = std::make_shared<InstructionBlock>();
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
    IB->Add(std::make_shared< InstructionReturn>(deps.DepthFirst.back()->Name()));

    // IB->EmitCode(std::cout);

    std::vector< std::shared_ptr<InstructionDeclareVariable> > decl_list;

    std::vector<std::shared_ptr< ImpliedMatrixFunction >> matrix_func_list;

    std::unordered_map<std::string, size_t> head_alloc_map = { {"x",0}, {"y",1} };
    std::vector< std::unordered_map<std::string, size_t> > allocation_map_list{ head_alloc_map };

    
   

    for (auto const& instr : *IB)
    {

        auto alloc_map = allocation_map_list.back();
        
        if (auto decl_instr = std::dynamic_pointer_cast<InstructionDeclareVariable>(instr))
        {
            decl_list.push_back(decl_instr);

            std::vector<std::string> comment_vec;

            auto expr = decl_instr->as_operator_();
            auto deps = expr->DepthFirstAnySymbolicDependencyOrThisNoRecurse();
            size_t index = 0;

            auto matrix_func = ImpliedMatrixFunction::Make(matrix_func_list.size(), instr);

            for (auto const& sym : deps.Set) {
                std::stringstream ss;
                ss << sym->Name() << " => ";
                auto diff = expr->Diff(sym->Name());
                diff->EmitCode(ss);
                std::string comment = ss.str();
                if (comment.back() == '\n')
                {
                    comment.pop_back();
                }
                comment_vec.push_back(comment);


            }

            alloc_map[decl_instr->LValueName()] = alloc_map.size();

            AADIB->Add(std::make_shared<InstructionComment>(comment_vec));
            AADIB->Add(matrix_func->MakeComment());
            AADIB->Add(decl_instr);

            matrix_func_list.push_back(matrix_func);
            allocation_map_list.push_back(alloc_map);

            
        } 
        else
        {
            // default
            AADIB->Add(instr);
        }

    }

    AADIB->Add(std::make_shared<InstructionComment>(std::vector<std::string>{ "//////////////", "Starting AAD matrix", "//////////////", }));
    std::vector<std::string> matrix_lvalues;
    for (size_t idx = matrix_func_list.size(); idx != 0; )
    {
        bool is_terminal = (idx == matrix_func_list.size());
        --idx;
        auto const& alloc_map = allocation_map_list[idx];
        AADIB->Add(matrix_func_list[idx]->MakeComment());

        auto matrix_decl = matrix_func_list[idx]->MakeMatrix(alloc_map, is_terminal);
        matrix_lvalues.push_back(matrix_decl->LValueName());
        AADIB->Add(matrix_decl);
    }
    
    std::string head = matrix_lvalues[0];
    for (size_t idx = 1; idx < matrix_lvalues.size(); ++idx)
    {
        head = "(" + matrix_lvalues[idx] + "*" + head + ")";
    }

    AADIB->Add(std::make_shared< InstructionText>("auto D = " + head + ";"));

    AADIB->Add(std::make_shared< InstructionText>("if( d_x) { *d_x = D[0]; }"));
    AADIB->Add(std::make_shared< InstructionText>("if( d_y) { *d_y = D[1]; }"));
    


    std::cout << "double f(double x, double y, double* d_x=nullptr, double* d_y = nullptr){\n";
    for (auto const& instr : *AADIB)
    {
        instr->EmitCode(std::cout);
        std::cout << "\n";
    }
    std::cout << "}\n";


    return 0;

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
    auto expr_deps = single_expr->DepthFirstAnySymbolicDependencyAndThis();

    std::unordered_map<std::string, std::shared_ptr<Operator> > adj_expr;
    for (auto head : expr_deps.DepthFirst) {
        adj_expr[head->Name()] = Constant::Make(0.0);
        std::cout << head->Name() << "\n";
    }
    std::cout << "----------------\n";

    return 0;
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
    }
   
}

#include <Eigen/Dense>

double f(double x, double y, double* d_x = nullptr, double* d_y = nullptr) {
    // x => 1

    // Matrix function F_0 => __symbol_4
    //     x => 1

    double const __symbol_4 = x;

    // __symbol_4 => ((((1)*(__symbol_4)))+(((__symbol_4)*(1))))

    // Matrix function F_1 => __symbol_5
    //     __symbol_4 => ((__symbol_4)+(__symbol_4))

    double const __symbol_5 = ((__symbol_4) * (__symbol_4));

    // y => 1

    // Matrix function F_2 => __symbol_1
    //     y => 1

    double const __symbol_1 = y;

    // __symbol_1 => ((((1)*(__symbol_1)))+(((__symbol_1)*(1))))

    // Matrix function F_3 => __symbol_2
    //     __symbol_1 => ((__symbol_1)+(__symbol_1))

    double const __symbol_2 = ((__symbol_1) * (__symbol_1));

    // __symbol_2 => ((((1)*(__symbol_1)))+(((__symbol_2)*(0))))
    // __symbol_1 => ((((0)*(__symbol_1)))+(((__symbol_2)*(1))))

    // Matrix function F_4 => __symbol_3
    //     __symbol_2 => __symbol_1
    //     __symbol_1 => __symbol_2

    double const __symbol_3 = ((__symbol_2) * (__symbol_1));

    // __symbol_5 => ((1)+(0))
    // __symbol_3 => ((0)+(1))

    // Matrix function F_5 => __symbol_6
    //     __symbol_5 => 1
    //     __symbol_3 => 1

    double const __symbol_6 = ((__symbol_5)+(__symbol_3));

    // __symbol_6 => 1

    // Matrix function F_6 => __statement_0
    //     __symbol_6 => 1

    double const __statement_0 = __symbol_6;

    // //////////////
    // Starting AAD matrix
    // //////////////

    // Matrix function F_6 => __statement_0
    //     __symbol_6 => 1

    const Eigen::Matrix<double, 8, 1>adj_matrix_6{
{0}
, {0}
, {0}
, {0}
, {0}
, {0}
, {0}
, {1}
    };

    // Matrix function F_5 => __symbol_6
    //     __symbol_5 => 1
    //     __symbol_3 => 1

    const Eigen::Matrix<double, 7, 8>adj_matrix_5{
{1, 0, 0, 0, 0, 0, 0, 0}
, {0, 1, 0, 0, 0, 0, 0, 0}
, {0, 0, 1, 0, 0, 0, 0, 0}
, {0, 0, 0, 1, 0, 0, 0, 1}
, {0, 0, 0, 0, 1, 0, 0, 0}
, {0, 0, 0, 0, 0, 1, 0, 0}
, {0, 0, 0, 0, 0, 0, 1, 1}
    };

    // Matrix function F_4 => __symbol_3
    //     __symbol_2 => __symbol_1
    //     __symbol_1 => __symbol_2

    const Eigen::Matrix<double, 6, 7>adj_matrix_4{
{1, 0, 0, 0, 0, 0, 0}
, {0, 1, 0, 0, 0, 0, 0}
, {0, 0, 1, 0, 0, 0, 0}
, {0, 0, 0, 1, 0, 0, 0}
, {0, 0, 0, 0, 1, 0, __symbol_2}
, {0, 0, 0, 0, 0, 1, __symbol_1}
    };

    // Matrix function F_3 => __symbol_2
    //     __symbol_1 => ((__symbol_1)+(__symbol_1))

    const Eigen::Matrix<double, 5, 6>adj_matrix_3{
{1, 0, 0, 0, 0, 0}
, {0, 1, 0, 0, 0, 0}
, {0, 0, 1, 0, 0, 0}
, {0, 0, 0, 1, 0, 0}
, {0, 0, 0, 0, 1, ((__symbol_1)+(__symbol_1))}
    };

    // Matrix function F_2 => __symbol_1
    //     y => 1

    const Eigen::Matrix<double, 4, 5>adj_matrix_2{
{1, 0, 0, 0, 0}
, {0, 1, 0, 0, 1}
, {0, 0, 1, 0, 0}
, {0, 0, 0, 1, 0}
    };

    // Matrix function F_1 => __symbol_5
    //     __symbol_4 => ((__symbol_4)+(__symbol_4))

    const Eigen::Matrix<double, 3, 4>adj_matrix_1{
{1, 0, 0, 0}
, {0, 1, 0, 0}
, {0, 0, 1, ((__symbol_4)+(__symbol_4))}
    };

    // Matrix function F_0 => __symbol_4
    //     x => 1

    const Eigen::Matrix<double, 2, 3>adj_matrix_0{
{1, 0, 1}
, {0, 1, 0}
    };

    auto D = (adj_matrix_0 * (adj_matrix_1 * (adj_matrix_2 * (adj_matrix_3 * (adj_matrix_4 * (adj_matrix_5 * adj_matrix_6))))));

    if (d_x) { *d_x = D[0]; }

    if (d_y) { *d_y = D[1]; }

    return __statement_0;

}

int main() {
    double x = 2.2;
    double y = 3.3;

    double d_x = 0;
    double d_y = 0;

    std::cout << "f=" << f(x, y, &d_x, &d_y) << "\n";
    std::cout << "d_x = " << d_x << "\n";
    std::cout << "d_y = " << d_y << "\n";


    if (true)
    {
        double e = 0.00001;
        std::cout << "d_x=" << ((f(x + e, y) - f(x - e, y)) / 2 / e) << "\n";
        std::cout << "d_y=" << ((f(x, y + e) - f(x, y - e)) / 2 / e) << "\n";
    }
    return 0;
}