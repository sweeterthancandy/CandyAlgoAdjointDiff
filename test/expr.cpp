#include <gtest/gtest.h>
#include <list>
#include "Cady/Cady.h"
#include "Cady/Transform.h"

using namespace Cady;

TEST(Expr,Constant){
        auto value = 1.234;
        auto x = Constant::Make(value);
        auto dx = x->Diff("x");

        EXPECT_EQ(OPKind_Constant, x->Kind()  );
        EXPECT_EQ(value, std::reinterpret_pointer_cast<Constant>(x)->Value() );
        EXPECT_EQ(OPKind_Constant, dx->Kind()  );
        EXPECT_EQ(0.0, std::reinterpret_pointer_cast<Constant>(dx)->Value() );
}

TEST(Expr,ExogenousSymbol){
        auto sym = "Foo";
        auto s = ExogenousSymbol::Make(sym);
        EXPECT_EQ(OPKind_ExogenousSymbol, s->Kind()  );
        EXPECT_EQ(sym, std::reinterpret_pointer_cast<ExogenousSymbol>(s)->Name() );

        auto ds_ds = s->Diff(sym);
        auto ds_dt = s->Diff("Other");

        EXPECT_EQ(OPKind_Constant, ds_ds->Kind());
        EXPECT_EQ(1.0, std::reinterpret_pointer_cast<Constant>(ds_ds)->Value() );
        EXPECT_EQ(OPKind_Constant, ds_dt->Kind());
        EXPECT_EQ(0.0, std::reinterpret_pointer_cast<Constant>(ds_dt)->Value() );
}

TEST(Expr,EndgenousSymbol){

        auto sub_expr = Constant::Make(2.0);

        auto sym = "Foo";
        auto s = EndgenousSymbol::Make(sym, sub_expr);
        EXPECT_EQ(OPKind_EndgenousSymbol, s->Kind()  );
        EXPECT_EQ(sym, std::reinterpret_pointer_cast<EndgenousSymbol>(s)->Name() );

        auto ds_ds = s->Diff(sym);
        auto ds_dt = s->Diff("Other");

        EXPECT_EQ(OPKind_Constant, ds_ds->Kind());
        EXPECT_EQ(1.0, std::reinterpret_pointer_cast<Constant>(ds_ds)->Value() );
        EXPECT_EQ(OPKind_Constant, ds_dt->Kind());
        EXPECT_EQ(0.0, std::reinterpret_pointer_cast<Constant>(ds_dt)->Value() );
}



enum InstructionKind{
        Instr_VarDecl,
        Instr_PointerAssignment,
        Instr_Return,
        Instr_Comment,
};
struct Instruction{
        explicit Instruction(InstructionKind kind)
                :kind_(kind)
        {}
        virtual ~Instruction()=default;
        InstructionKind Kind()const{ return kind_; }
        virtual void EmitCode(std::ostream& out)const=0;
private:
        InstructionKind kind_;
};
struct InstructionComment : Instruction{
        InstructionComment(std::vector<std::string> const& text)
                :Instruction{Instr_Comment}
                ,text_{text}
        {}
        virtual void EmitCode(std::ostream& out)const{
                for(auto const& comment : text_){
                        out << "    // " << comment << "\n";
                }
        }
private:
        std::vector<std::string> text_;
};
struct InstructionDeclareVariable : Instruction{
        InstructionDeclareVariable(std::string const& name, std::shared_ptr<Operator> op)
                :Instruction{Instr_VarDecl}
                ,name_(name)
                ,op_(op)
        {}
        virtual void EmitCode(std::ostream& out)const{
                out << "    double const " << name_ << " = ";
                op_->EmitCode(out);
                out << ";\n";
        }
        auto const& as_operator_()const{ return op_; }
private:
        std::string name_;
        std::shared_ptr<Operator> op_;
};
struct InstructionPointerAssignment : Instruction{
        InstructionPointerAssignment(std::string const& name, std::string const& r_value)
                :Instruction{Instr_PointerAssignment}
                ,name_(name)
                ,r_value_{r_value}
        {}
        virtual void EmitCode(std::ostream& out)const{
                out << "    *" << name_ << " = " << r_value_ << ";\n";
        }
private:
        std::string name_;
        std::string r_value_;
};
struct InstructionReturn : Instruction{
        InstructionReturn(std::string const& name)
                :Instruction{Instr_Return}
                ,name_(name)
        {}
        virtual void EmitCode(std::ostream& out)const{
                out << "    return " << name_ << ";\n";
        }
private:
        std::string name_;
};

struct InstructionBlock : std::vector<std::shared_ptr<Instruction> >{
        virtual ~InstructionBlock()=default;
        void Add(std::shared_ptr<Instruction> instr){
                // quick hack for now
                if( size() > 0 && back()->Kind() == Instr_Return ){
                        auto tmp = back();
                        pop_back();
                        push_back(instr);
                        push_back(tmp);
                } else {
                        push_back(instr);
                }
        }
        virtual void EmitCode(std::ostream& out)const{
                for(auto const& ptr : *this){
                        ptr->EmitCode(out);
                }
        }
};

TEST(Instruction,InstructionDeclareVariable){
        auto expr = BinaryOperator::Add(
                Constant::Make(2.0),
                Constant::Make(3.0)
        );
        auto instr = std::make_shared<InstructionDeclareVariable>("tmp", expr);
        instr->EmitCode(std::cout);

}

/*
        From Savine
 */
TEST(Black,Numeric){
        auto S0  = ExogenousSymbol::Make("S0");
        auto r   = ExogenousSymbol::Make("r");
        auto y   = ExogenousSymbol::Make("y");
        auto sig = ExogenousSymbol::Make("sig");
        auto K   = ExogenousSymbol::Make("K");
        auto T   = ExogenousSymbol::Make("T");

        auto df = Exp::Make( 
                UnaryOperator::UnaryMinus(
                        BinaryOperator::Mul(
                                r,
                                T
                        )
                )
        );
        auto sym_df = EndgenousSymbol::Make("df", df);

        auto F = BinaryOperator::Mul(
                S0,
                Exp::Make(
                        BinaryOperator::Mul(
                                BinaryOperator::Sub(r,y),
                                T
                        )
                )
        );
        auto sym_F = EndgenousSymbol::Make("F", F);

        auto std = BinaryOperator::Mul(
                sig,
                BinaryOperator::Pow(
                        T,
                        Constant::Make(0.5)
                )
        );
        auto sym_std = EndgenousSymbol::Make("std", std);

        auto d = BinaryOperator::Div(
                Log::Make(
                        BinaryOperator::Div(
                                sym_F,
                                K
                        )
                ),
                sym_std
        );
        auto sym_d = EndgenousSymbol::Make("d", d);

        auto d1 = BinaryOperator::Add(
                sym_d,
                BinaryOperator::Mul(
                        Constant::Make(0.5),
                        sym_std
                )
        );
        auto sym_d1 = EndgenousSymbol::Make("d1", d1);
        
        auto d2 = BinaryOperator::Sub(
                sym_d,
                BinaryOperator::Mul(
                        Constant::Make(0.5),
                        sym_std
                )
        );
        auto sym_d2 = EndgenousSymbol::Make("d2", d2);

        auto nd1 = Phi::Make(sym_d1);
        auto sym_nd1 = EndgenousSymbol::Make("nd1", nd1);
        
        auto nd2 = Phi::Make(sym_d2);
        auto sym_nd2 = EndgenousSymbol::Make("nd2", nd2);

        auto c = BinaryOperator::Mul(
                sym_df,
                BinaryOperator::Sub(
                        BinaryOperator::Mul(
                                sym_F,
                                sym_nd1
                        ),
                        BinaryOperator::Mul(
                                K,
                                sym_nd2
                        )
                )
        );
        auto sym_c = EndgenousSymbol::Make("c", c);

        double S0_value = 10;
        double r_value = 0.05;
        double y_value = 0.00;
        double sig_value = 0.2;
        double K_value = 13;
        double T_value = 1;

        SymbolTable ST;
        ST("S0" , S0_value);
        ST("r"  , r_value);
        ST("y"  , y_value);
        ST("sig", sig_value);
        ST("K"  , K_value);
        ST("T"  , T_value);

        std::cout << "c->Eval(ST) => " << c->Eval(ST) << "\n"; // __CandyPrint__(cxx-print-scalar,c->Eval(ST))

        using std::exp;
        using std::sqrt;
        auto phi = [](auto x){
                return std::erfc(-x/std::sqrt(2))/2;
        };

        double df_value = exp(-r_value*T_value);
        EXPECT_FLOAT_EQ(df_value, df->Eval(ST));

        double F_value = S0_value * exp((r_value-y_value)*T_value);
        EXPECT_FLOAT_EQ(F_value, F->Eval(ST));

        double std_value = sig_value * sqrt(T_value);
        EXPECT_FLOAT_EQ(std_value, std->Eval(ST));
        
        double d_value = log(F_value/K_value)/std_value;
        EXPECT_FLOAT_EQ(d_value, d->Eval(ST));

        double d1_value = d_value + 0.5 * std_value;
        EXPECT_FLOAT_EQ(d1_value, d1->Eval(ST));

        double d2_value = d_value - 0.5 * std_value;
        EXPECT_FLOAT_EQ(d2_value, d2->Eval(ST));

        double nd1_value = phi(d1_value);
        EXPECT_FLOAT_EQ(nd1_value, nd1->Eval(ST));

        double nd2_value = phi(d2_value);
        EXPECT_FLOAT_EQ(nd2_value, nd2->Eval(ST));

        double c_value = df_value * ( F_value * nd1_value - K_value * nd2_value );
        EXPECT_FLOAT_EQ(c_value, c->Eval(ST));
        
        struct RemoveEndo : OperatorTransform{
                virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr){
                        auto candidate = ptr->Clone(shared_from_this());
                        if( candidate->Kind() == OPKind_EndgenousSymbol ){
                                if( auto typed = std::dynamic_pointer_cast<EndgenousSymbol>(candidate)){
                                        return typed->Expr();
                                }
                        }
                        return candidate;
                }
        };
        auto c_raw = c->Clone(std::make_shared<RemoveEndo>());

        EXPECT_FLOAT_EQ(c->Eval(ST), c_raw->Eval(ST));

        double epsilon = 0.00001;
        auto baseline = c->Eval(ST);
        auto bump_ST = ST;
        bump_ST("sig", sig_value + epsilon);
        auto sig_bump = c->Eval(bump_ST);
        auto fd_d_sig = ( sig_bump - baseline ) / epsilon;
        std::cout << "sig_bump => " << sig_bump << "\n"; // __CandyPrint__(cxx-print-scalar,sig_bump)
        std::cout << "fd_d_sig => " << fd_d_sig << "\n"; // __CandyPrint__(cxx-print-scalar,fd_d_sig)
        std::cout << "c_raw->Diff(\"sig\")->Eval(ST) => " << c_raw->Diff("sig")->Eval(ST) << "\n"; // __CandyPrint__(cxx-print-scalar,c_raw->Diff("sig")->Eval(ST))


        double adj_S0  = 0.0;
        double adj_r   = 0.0;
        double adj_y   = 0.0;
        double adj_sig = 0.0;
        double adj_K   = 0.0;
        double adj_T   = 0.0;
        double adj_df  = 0.0;
        double adj_F   = 0.0;
        double adj_std = 0.0;
        double adj_d   = 0.0;
        double adj_d1  = 0.0;
        double adj_d2  = 0.0;
        double adj_nd1 = 0.0;
        double adj_nd2 = 0.0;
        double adj_c   = 1.0;

        EXPECT_FLOAT_EQ(F_value * nd1_value - K_value * nd2_value, c->Diff("df")->Eval(ST));
        EXPECT_FLOAT_EQ(df_value * nd1_value, c->Diff("F")->Eval(ST));
        EXPECT_FLOAT_EQ(df_value * F_value, c->Diff("nd1")->Eval(ST));
        EXPECT_FLOAT_EQ(-df_value * nd2_value, c->Diff("K")->Eval(ST));
        EXPECT_FLOAT_EQ(-df_value * K_value, c->Diff("nd2")->Eval(ST));

        adj_df   +=  adj_c * c->Diff("df")->Eval(ST);
        adj_F    +=  adj_c * c->Diff("F") ->Eval(ST);
        adj_nd1  +=  adj_c * c->Diff("nd1") ->Eval(ST);
        adj_K    +=  adj_c * c->Diff("K") ->Eval(ST);
        adj_nd2  +=  adj_c * c->Diff("nd2") ->Eval(ST);

        adj_d1   += adj_nd1 * nd1->Diff("d1")->Eval(ST);
        adj_d2   += adj_nd2 * nd2->Diff("d2")->Eval(ST);

        adj_d    += adj_d1 * d1->Diff("d1")->Eval(ST);
        adj_std  += adj_d1 * d1->Diff("std")->Eval(ST);
        adj_d    += adj_d2 * d2->Diff("d2")->Eval(ST);
        adj_std  += adj_d2 * d2->Diff("std")->Eval(ST);

        adj_F    += adj_d * d->Diff("F")->Eval(ST);
        adj_K    += adj_d * d->Diff("K")->Eval(ST);
        adj_std  += adj_d * d->Diff("std")->Eval(ST);

        adj_sig  += adj_std * std->Diff("sig")->Eval(ST);

        std::cout << "adj_sig => " << adj_sig << "\n"; // __CandyPrint__(cxx-print-scalar,adj_sig)


        #if 0

        struct StackFrame{
                std::string name;
                std::shared_ptr<Operator> Op;
        };
        std::list<StackFrame> stack{StackFrame{"c", c}};

        std::unordered_set<std::string> seen;
        for(;stack.size();){
                auto head = stack.front();
                stack.pop_front();
                auto deps = head.Op->DepthFirstAnySymbolicDependencyNoRecurse();
                for(auto ptr : deps.Set){
                        if(ptr->IsEndo() ){
                                if( seen.count(ptr->Name()) == 0){
                                        std::cout << head.name << " -> " << ptr->Name() << "\n"; // __CandyPrint__(cxx-print-scalar,ptr->Name())
                                        stack.push_back(StackFrame{ptr->Name(), std::reinterpret_pointer_cast<EndgenousSymbol>(ptr)->Expr()});
                                        seen.insert(ptr->Name());
                                }
                        } else {
                                //std::cout << head.name << " -> " << ptr->Name() << " [terminal]\n"; // __CandyPrint__(cxx-print-scalar,ptr->Name())
                        }
                }
        }

        std::cout << "\n\n";
        #endif
        auto deps = sym_c->DepthFirstAnySymbolicDependencyAndThis();

        for(auto ptr : deps.DepthFirst ){
                if( ptr->IsExo() )
                        continue;
                std::cout << "ptr->Name() => " << ptr->Name() << "\n"; // __CandyPrint__(cxx-print-scalar,ptr->Name())
        }


        std::unordered_map<std::string, double> adj;
        std::unordered_map<std::string, std::shared_ptr<Operator> > adj_expr;
        adj[deps.DepthFirst.back()->Name()] = 1.0; // seed

        for(auto head : deps.DepthFirst){
                adj_expr[head->Name()] = Constant::Make(0.0);
        }
        adj_expr[deps.DepthFirst.back()->Name()] = Constant::Make(1.0);
        
        for(size_t idx=deps.DepthFirst.size();idx!=0;){
                --idx;
                auto head = deps.DepthFirst[idx];
                if( head->IsExo() )
                        continue;
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                auto subs = head->DepthFirstAnySymbolicDependencyNoRecurse();
                for(auto ptr : subs.DepthFirst){
                        std::cout << "adj_" << ptr->Name() << " => " << "adj_" << head->Name() << " * " << expr->Diff(ptr->Name())->Eval(ST) << "\n";
                        adj[ptr->Name()] += adj[head->Name()]*expr->Diff(ptr->Name())->Eval(ST);

                        adj_expr[ptr->Name()] = BinaryOperator::Add(
                                adj_expr[ptr->Name()],
                                BinaryOperator::Mul(
                                        adj_expr[head->Name()],
                                        expr->Diff(ptr->Name())
                                )
                        );

                }
        }
        for(auto const& p : adj){
                std::cout << std::left << std::setw(10) << p.first << " => " << p.second << "\n";
        }

        auto adj_diff = [&](auto sym){
                return adj[sym];
        };
        auto fd_diff = [&](auto sym){
                auto bump_ST = ST;
                bump_ST(sym, ST[sym] + epsilon);
                auto sig_bump = c->Eval(bump_ST);
                return ( sig_bump - baseline ) / epsilon;
        };
        auto ana_diff = [&](auto sym){
                return c_raw->Diff(sym)->Eval(ST);
        };


        std::vector<std::string> exo{ "S0", "r", "y", "sig", "K", "T"};
        for(auto sym : exo ){
                std::cout << "adj_diff(sym) => " << adj_diff(sym) << "\n"; // __CandyPrint__(cxx-print-scalar,adj_diff(sym))
                std::cout << "fd_diff(sym) => " << fd_diff(sym) << "\n"; // __CandyPrint__(cxx-print-scalar,fd_diff(sym))
                std::cout << "ana_diff(sym) => " << ana_diff(sym) << "\n"; // __CandyPrint__(cxx-print-scalar,ana_diff(sym))

                EXPECT_FLOAT_EQ( ana_diff(sym), adj_diff(sym));
        }

        InstructionBlock IB;
        for(auto head : deps.DepthFirst){
                if( head->IsExo() )
                        continue;
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                auto subs = head->DepthFirstAnySymbolicDependencyNoRecurse();
                IB.Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
        }

        Transform::FoldZero fold_zero;
        for(auto p : adj_expr ){
                auto folded = fold_zero.Fold(p.second);
                IB.Add(std::make_shared<InstructionDeclareVariable>("adj_" + p.first, folded));
        }
        IB.Add(std::make_shared<InstructionReturn>("c"));
        IB.EmitCode(std::cout);

}


#include "Cady/Frontend.h"
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


TEST(Kernel,Black){
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

        InstructionBlock IB;
        auto deps = expr->DepthFirstAnySymbolicDependencyAndThis();
        for(auto head : deps.DepthFirst){
                if( head->IsExo() )
                        continue;
                auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                auto subs = head->DepthFirstAnySymbolicDependencyNoRecurse();
                IB.Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
        }
        IB.Add(std::make_shared<InstructionReturn>(expr->Name()));
        IB.EmitCode(std::cout);

        std::cout << "expr->Kind() => " << expr->Kind() << "\n"; // __CandyPrint__(cxx-print-scalar,expr->Kind())
}

/*
        struct BaseName{
                struct ResultType{
                        double Fields[0];
                        double Fields[1];
                        ...
                        double Fields[n-1];
                };
                virtual ~BaseBame()=default;
                virtual ResultType FunctionType(double Args[0],
                                                double Args[1],
                                                ...
                                                double Args[m-1])const=0;
                using map_ty = std::unordered_map<std::string, std::shared_ptr<BaseName> >;
                static map_ty& AllImplementations(){
                        static map_ty;
                        return map_ty;
                }
        };

 */
struct ProfileFunctionDefinition{
        std::string BaseName;
        std::string FunctionName;
        std::vector<std::string> Args;
        std::vector<std::string> Fields;
};

struct ProfileFunction{
        std::string ImplName;
        std::shared_ptr<InstructionBlock> IB;
        std::unordered_map<std::string, std::string> ResultMapping;
};

struct ProfileCodeGen{
        explicit ProfileCodeGen(std::shared_ptr<ProfileFunctionDefinition> const& def):def_{def}{}
        void AddImplementation(std::shared_ptr<ProfileFunction> func){
                vec_.push_back(func);
        }
        void EmitCode(std::ostream& out = std::cout){
                out << "#include <vector>\n";
                out << "#include <unordered_map>\n";
                out << "#include <cstdio>\n";
                out << "#include <cmath>\n";
                out << "#include <iostream>\n";
                out << "#include <memory>\n";
                out << "#include <boost/timer/timer.hpp>\n";
                out << "#include <boost/lexical_cast.hpp>\n";
                out << "\n";
                out << "\n";
                out << "struct " << def_->BaseName << "{\n";
                out << "    struct ResultType{\n";
                for(auto const& field : def_->Fields ){
                out << "        double " << field << " = 0.0;\n";
                };
                out << "    };\n";
                out << "    virtual ~" << def_->BaseName << "()=default;\n";
                out << "    virtual ResultType " << def_->FunctionName << "(\n";
                for(size_t idx=0;idx!=def_->Args.size();++idx){
                out << "        " << (idx==0?"  ":", ") << "double " << def_->Args[idx] << (idx+1==def_->Args.size()?")const=0;\n":"\n");
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


                for(auto func : vec_){
                out << "struct " << func->ImplName << " : " << def_->BaseName << "{\n";
                out << "    virtual ResultType " << def_->FunctionName << "(\n";
                for(size_t idx=0;idx!=def_->Args.size();++idx){
                out << "        " << (idx==0?"  ":", ") << "double " << def_->Args[idx] << (idx+1==def_->Args.size()?")const override{\n":"\n");
                }
                func->IB->EmitCode(out);
                out << "        ResultType result;\n";
                for(auto const& p : func->ResultMapping){
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
                out << "    auto impls = " << def_->BaseName << "::AllImplementations();\n";
                out << "    for(volatile size_t N = 100;;N*=2){\n";
                out << "        std::vector<std::string> line;\n";
                out << "        line.push_back(boost::lexical_cast<std::string>(N));\n";
                out << "        for(auto impl : impls ){\n";
                out << "            boost::timer::cpu_timer timer;\n";
                out << "            for(volatile size_t idx=0;idx!=N;++idx){\n";
                out << "                auto result = impl.second->" << def_->FunctionName << "(t,T,r,S,K,vol);\n";
                out << "            }\n";
                out << "            line.push_back(timer.format(4, \"%w\"));\n";
                out << "        }\n";
                out << "        for(auto item : line){ std::cout << item << ','; } std::cout << '\\n';\n";
                out << "    }\n";
                out << "}\n";


        }
private:
        std::shared_ptr<ProfileFunctionDefinition> def_;
        std::vector<std::shared_ptr<ProfileFunction> > vec_;
};


struct NaiveBlackProfile : ProfileFunction{
        NaiveBlackProfile(){
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
                for(auto head : deps.DepthFirst){
                        if( head->IsExo() )
                                continue;
                        auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                        IB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                }
                ResultMapping["c"] = "__statement_7";
        }
};

TEST(ProfileGen,A){
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

        
        cg.EmitCode();

        std::ofstream out("pc.cxx");
        cg.EmitCode(out);
}



