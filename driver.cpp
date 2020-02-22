#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>

#include <boost/optional.hpp>
#include <boost/variant.hpp>
#


#if 0
struct SymbolTable{
private:
        std::unordered_map<std::string, double> m_;
};
#endif

struct Operator : std::enable_shared_from_this<Operator>{

        explicit Operator(std::string const& name)
                : name_{name}
        {}
        virtual ~Operator(){}


        #if 0
        virtual double Eval(SymbolTable const& ST)const=0;
        #endif
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const=0;
        virtual void EmitCode(std::ostream& ss)const=0;
        
        size_t Arity()const{ return children_.size(); }
        std::shared_ptr<Operator> At(size_t idx)const{
                if( Arity() < idx ){
                        throw std::domain_error("getting child that doesn't exist");
                }
                return children_.at(idx);
        }
        auto&       Children()     { return children_; }
        auto const& Children()const{ return children_; }

        bool IsTerminal()const{ return Arity() == 0; }
        bool IsNonTerminal()const{ return Arity() > 0; }

        std::string const& Name()const{ return name_; }

        virtual std::vector<std::string> HiddenArguments()const{ return {}; }

        void Display(std::ostream& ostr = std::cout)const{

                struct EndOfGroup{};
                using ptr_t = std::shared_ptr<Operator const>;
                using var_t = boost::variant<
                        ptr_t ,
                        EndOfGroup
                >;
                std::vector<var_t> stack{var_t{shared_from_this()}};

                auto indent = [&](int extra = 0){
                        return std::string((stack.size()+extra)*2,' ');
                };

                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        if( auto opt_ptr = boost::get<ptr_t>(&head) ){
                                auto ptr = *opt_ptr;

                                auto hidden = ptr->HiddenArguments();
                                if( ptr->IsTerminal() ){
                                        if( hidden.size() == 0 ){
                                                ostr << indent() << ptr->Name() << "{}\n";
                                        } else if( hidden.size() == 1 ){
                                                ostr << indent() << ptr->Name() << "{" << hidden[0] << "}\n";
                                        } else {
                                                ostr << indent() << ptr->Name() << "{\n";
                                                for(auto const& s : hidden ){
                                                        ostr << indent(1) << s << "\n";
                                                }
                                                ostr << indent() << "}\n";
                                        }
                                } else {
                                        ostr << indent() << ptr->Name() << "{\n";
                                        stack.push_back(EndOfGroup{});
                                        auto children = ptr->Children();
                                        for(auto iter = children.rbegin(), end = children.rend();iter!=end;++iter){
                                                stack.push_back(var_t{*iter});
                                        }
                                }
                        } else if( auto ptr = boost::get<ptr_t>(&head)){
                                ostr << indent() << "}\n";
                        }
                }
        }
protected:
        size_t Push(std::shared_ptr<Operator> const& ptr){
                size_t slot = children_.size();
                children_.push_back(ptr);
                return slot;
        }
private:
        std::string name_;
        std::vector<std::shared_ptr<Operator> > children_;

};


struct Constant : Operator{
        Constant(double value)
                :Operator{"Constant"}
                ,value_(value)
        {}
        #if 0
        virtual double Eval(SymbolTable const& ST)const{
                return value_;
        }
        #endif
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                return Constant::Make(0.0);
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << value_;
        }

        static std::shared_ptr<Operator> Make(double value){
                return std::make_shared<Constant>(value);
        }
        virtual std::vector<std::string> HiddenArguments()const{ return { std::to_string(value_) }; }
private:
        double value_;
};

struct Symbol : Operator{
        Symbol(std::string const& name)
                :Operator{"Symbol"}
                ,name_(name)
        {}
        #if 0
        virtual double Eval(SymbolTable const& ST)const{
                return ST.Lookup(name_);
        }
        #endif
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                if( symbol == name_ ){
                        return Constant::Make(1.0);
                }
                return Constant::Make(0.0);
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << name_;
        }
        std::string const& Name()const{ return name_; }
        
        static std::shared_ptr<Symbol> Make(std::string const& symbol){
                return std::make_shared<Symbol>(symbol);
        }
private:
        std::string name_;
};

enum BinaryOperatorKind{
        OP_ADD,
        OP_SUB,
        OP_MUL,
        OP_DIV,
        OP_POW,
};

struct BinaryOperator : Operator{
        BinaryOperator(BinaryOperatorKind op, std::shared_ptr<Operator> left, std::shared_ptr<Operator> right)
                :Operator{"BinaryOperator"}
                ,op_(op)
                        #if 0
                        ,left_(left)
                        ,right_(right)
                        #endif
        {
                Push(left);
                Push(right);
        }
        std::shared_ptr<Operator> LParam()const{ return At(0); }
        std::shared_ptr<Operator> RParam()const{ return At(1); }
        #if 0
        virtual double Eval(SymbolTable const& ST)const{
                switch(op_)
                {
                case OP_ADD:
                        {
                                return left_->Eval(ST) + RParam()->Eval(ST);
                        }
                case OP_SUB:
                        {
                                return left_->Eval(ST) - RParam()->Eval(ST);
                        }
                case OP_MUL:
                        {
                                return left_->Eval(ST) * RParam()->Eval(ST);
                        }
                case OP_DIV:
                        {
                                return left_->Eval(ST) / RParam()->Eval(ST);
                        }
                }
        }
        #endif
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const
        {
                switch(op_)
                {
                        case OP_ADD:
                        {
                                return Add(
                                        LParam()->Diff(symbol),
                                        RParam()->Diff(symbol)
                                );
                        }
                        case OP_SUB:
                        {
                                return Sub(
                                        LParam()->Diff(symbol),
                                        RParam()->Diff(symbol)
                                );
                        }
                        case OP_MUL:
                        {
                                return Add(
                                        Mul( LParam()->Diff(symbol), RParam()),
                                        Mul( LParam(), RParam()->Diff(symbol))
                                );
                        }
                        case OP_DIV:
                        {
                                return Div(
                                        Sub(
                                                Mul(
                                                        LParam()->Diff(symbol),
                                                        RParam()
                                                ),
                                                Mul(
                                                        LParam(),
                                                        RParam()->Diff(symbol)
                                                )
                                        ),
                                        Pow(
                                                RParam(),
                                                Constant::Make(2.0)
                                        )
                                );

                        }
                        case OP_POW:
                        {
                                // lets assume that the exponent is indpedent
                                // of the deriviative for now
                                //
                                // f(x)^C = C * f(x)*(C-1) * f'(x)
                                // ~ left ^ right
                                //
                                return Mul(
                                        RParam(),
                                        Mul(
                                                Pow(
                                                        LParam(),
                                                        Sub(
                                                                RParam(),
                                                                Constant::Make(1.0)
                                                        )
                                                ),
                                                LParam()->Diff(symbol)
                                        )
                                );
                        }
                }
        }


        virtual void EmitCode(std::ostream& ss)const{
                if( op_ == OP_POW ){
                        ss << "std::pow(";
                        LParam()->EmitCode(ss);
                        ss << ", ";
                        RParam()->EmitCode(ss);
                        ss << ")";
                } else {
                        ss << "(";
                        ss << "(";
                        LParam()->EmitCode(ss);
                        ss << ")";
                        switch(op_){
                        case OP_ADD: ss << "+"; break;
                        case OP_SUB: ss << "-"; break;
                        case OP_MUL: ss << "*"; break;
                        case OP_DIV: ss << "/"; break;
                        }
                        ss << "(";
                        RParam()->EmitCode(ss);
                        ss << ")";
                        ss << ")";
                }
        }


        static std::shared_ptr<Operator> Add(std::shared_ptr<Operator> const& left,
                                             std::shared_ptr<Operator> const& right)
        {
                return std::make_shared<BinaryOperator>(OP_ADD, left, right);
        }
        static std::shared_ptr<Operator> Sub(std::shared_ptr<Operator> const& left,
                                             std::shared_ptr<Operator> const& right)
        {
                return std::make_shared<BinaryOperator>(OP_SUB, left, right);
        }
        static std::shared_ptr<Operator> Mul(std::shared_ptr<Operator> const& left,
                                             std::shared_ptr<Operator> const& right)
        {
                return std::make_shared<BinaryOperator>(OP_MUL, left, right);
        }
        static std::shared_ptr<Operator> Div(std::shared_ptr<Operator> const& left,
                                             std::shared_ptr<Operator> const& right)
        {
                return std::make_shared<BinaryOperator>(OP_DIV, left, right);
        }
        static std::shared_ptr<Operator> Pow(std::shared_ptr<Operator> const& left,
                                             std::shared_ptr<Operator> const& right)
        {
                return std::make_shared<BinaryOperator>(OP_POW, left, right);
        }

private:
        BinaryOperatorKind op_;
        std::shared_ptr<Operator> left_;
        std::shared_ptr<Operator> right_;
};

struct Exp : Operator{
        Exp(std::shared_ptr<Operator> arg)
                :Operator{"Exp"}
                ,arg_(arg)
        {}
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                return BinaryOperator::Mul(
                        std::make_shared<Exp>(arg_),
                        arg_->Diff(symbol));
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << "std::exp(";
                arg_->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Exp> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Exp>(arg);
        }
private:
        std::shared_ptr<Operator> arg_;
};

struct Log : Operator{
        Log(std::shared_ptr<Operator> arg)
                :Operator{"Log"}
                ,arg_(arg)
        {}
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                return BinaryOperator::Div(
                        arg_->Diff(symbol),
                        arg_);
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << "std::log(";
                arg_->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Log> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Log>(arg);
        }
private:
        std::shared_ptr<Operator> arg_;
};


// normal distribution CFS
struct Phi : Operator{
        Phi(std::shared_ptr<Operator> arg)
                :Operator{"Phi"}
                ,arg_(arg)
        {}
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                // f(x) = 1/\sqrt{2 \pi} \exp{-\frac{1}{2}x^2}

                return 
                        BinaryOperator::Mul(
                                BinaryOperator::Div(
                                        Exp::Make(
                                                BinaryOperator::Sub(
                                                        Constant::Make(0.0),
                                                        BinaryOperator::Mul(
                                                                Constant::Make(0.5),
                                                                BinaryOperator::Pow(
                                                                        arg_,
                                                                        Constant::Make(2.0)
                                                                )
                                                        )
                                                )
                                        ),
                                        Constant::Make(2.506628274631000502415765284811045253006986740609938316629)
                                ),
                                arg_->Diff(symbol)
                        );

        }
        virtual void EmitCode(std::ostream& ss)const{
                // std::erfc(-x/std::sqrt(2))/2
                ss << "std::erfc(-(";
                arg_->EmitCode(ss);
                ss << ")/std::sqrt(2))/2";
        }

        static std::shared_ptr<Operator> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Phi>(arg);
        }

private:
        std::shared_ptr<Operator> arg_;
};

struct Statement : Symbol{
        Statement(std::string const& name, std::shared_ptr<Operator> expr)
                :Symbol(name),
                expr_(expr)
        {}
        std::shared_ptr<Operator> Expr()const{ return expr_; }
private:
        std::string name_;
        std::shared_ptr<Operator> expr_;
};

struct TemporaryAllocator{
        std::string Allocate(){
                std::stringstream ss;
                ss << prefix_ << index_;
                ++index_;
                return ss.str();
        }
private:
        std::string prefix_{"__temp_"};
        size_t index_{0};
};

struct Function{
        explicit Function(std::string const& name):name_{name}{}
        void AddArgument(std::string const& symbol){
                args_.push_back(symbol);
        }
        void AddStatement(std::shared_ptr<Statement> stmt){
                stmts_.push_back(stmt);
        }
        auto const& Arguments()const{ return args_; }
        auto const& Statements()const{ return stmts_; }
        std::string const& Name()const{ return name_; }

private:
        std::string name_;
        std::vector<std::string> args_;
        std::vector<std::shared_ptr<Statement> > stmts_;
};


struct StringCodeGenerator{
        void Emit(std::ostream& ss, Function const& f)const{

                // we have a vector [ x1, x2, ... ] which are the function 
                // parameters. 

                struct VariableInfo{
                        VariableInfo(std::string const& name)
                                : name_{name}
                        {}
                        std::string const& Name()const{ return name_; }
                        boost::optional<std::string const&> GetDiffLexical(std::string const& symbol)const{
                                auto iter = diff_map_.find(symbol);
                                if( iter == diff_map_.end() )
                                        return {};
                                return iter->second;
                        }
                        void MapDiff(std::string const& symbol, std::string const& name){
                                diff_map_[symbol] = name;
                        }
                private:
                        std::string name_;
                        std::unordered_map<std::string, std::string> diff_map_;
                };


                auto to_diff = f.Arguments();


                std::vector<std::shared_ptr<VariableInfo> > deps;
                for( auto const& arg : f.Arguments() ){
                        auto ptr = std::make_shared<VariableInfo>(arg);
                        for( auto const& inner_arg : to_diff ){
                                if( arg == inner_arg ){
                                        ptr->MapDiff(inner_arg, "1.0");
                                } else {
                                        ptr->MapDiff(inner_arg, "0.0");
                                }
                        }
                        deps.push_back(ptr);
                }



                ss << "double " << f.Name() << "(";
                for(size_t idx=0;idx!=deps.size();++idx){
                        if( idx != 0 ) 
                                ss << ", ";
                        ss << "double " << deps[idx]->Name();
                        ss << ", double* " << "d_" + deps[idx]->Name();
                }

                ss << ")\n";
                ss << "{\n";

                std::string indent = "    ";

                TemporaryAllocator temp_alloc;

                for(size_t idx=0;idx!=f.Statements().size();++idx){
                        // for each statement we need to add two calculations to the
                        // infomation
                        //      statement = expr
                        //      for each X in to-diff:
                        //        d_statement_X = D[X](expr)
                        //      
                        //    

                        //

                        auto const& stmt = f.Statements()[idx];
                        auto const& expr = stmt->Expr();
                        
                        auto stmt_dep = std::make_shared<VariableInfo>(stmt->Name());
                        
                        ss << indent << "/* expr\n";
                        expr->Display(ss);
                        ss << indent << "*/";
                        ss << indent << "double " << stmt_dep->Name() << " = ";
                        expr->EmitCode(ss);
                        ss << ";\n";


                        for( auto const& d_symbol : to_diff ){
                                std::vector<std::string> subs;

                                for( auto const& info : deps ){

                                        auto temp_name = temp_alloc.Allocate();

                                         ;

                                        // \partial stmt / \partial symbol d symbol
                                        auto sub_diff = BinaryOperator::Mul(
                                                expr->Diff( info->Name() ),
                                                Symbol::Make(*info->GetDiffLexical(d_symbol)));

                                        ss << indent << "// \\partial " << stmt->Name() << " / \\partial " << info->Name() << " d " << info->Name() << "\n";
                                        ss << indent << "double " << temp_name << " = ";
                                        sub_diff->EmitCode(ss);
                                        ss << ";\n";

                                        subs.push_back(temp_name);
                                }


                                std::string token = "__diff_" + stmt->Name() + "_" + d_symbol;
                                stmt_dep->MapDiff( d_symbol, token);


                                ss << indent << "double " << token << " = ";
                                for(size_t idx=0;idx!=subs.size();++idx){
                                        if( idx != 0 )
                                                ss << " + ";
                                        ss << subs[idx];
                                }
                                ss << ";\n";
                        }
                        ss << "\n\n\n";
                        deps.emplace_back(stmt_dep);

                }
                        
                for( auto const& d_symbol : to_diff ){
                        ss << indent << "*d_" + d_symbol << " = " << *deps.back()->GetDiffLexical(d_symbol) << ";\n";
                }

                ss << indent << "return " << deps.back()->Name() << ";\n";
                ss << "}\n";

        }
};

void example_0(){


        Function f("f");
        f.AddArgument("x");
        f.AddArgument("y");

        //auto expr_0 = BinaryOperator::Mul(Log::Make(BinaryOperator::Mul(Symbol::Make("x"),Symbol::Make("x"))),  Exp::Make(Symbol::Make("y")));
        //auto expr_0 = BinaryOperator::Pow(Symbol::Make("x"), Constant::Make(2));
        auto expr_0 = Phi::Make(BinaryOperator::Pow(Symbol::Make("x"), Constant::Make(3)));

        auto stmt_0 = std::make_shared<Statement>("stmt0", expr_0);

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

        auto expr_0 = BinaryOperator::Mul( Symbol::Make("a"), BinaryOperator::Mul(Symbol::Make("x"),  Symbol::Make("x")));


        auto stmt_0 = std::make_shared<Statement>("stmt0", expr_0);

        auto expr_1 = BinaryOperator::Add( Symbol::Make(stmt_0->Name()), Symbol::Make("b"));

        auto stmt_1 = std::make_shared<Statement>("stmt1", expr_1);
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
                Symbol::Make("T"),
                Symbol::Make("t")
        );

        auto deno = BinaryOperator::Mul( 
                Constant::Make(1.0),
                BinaryOperator::Mul(
                        Symbol::Make("vol"),
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
                                        Symbol::Make("S"),
                                        Symbol::Make("K")
                                )
                        ),
                        BinaryOperator::Mul(
                                BinaryOperator::Add(
                                        Symbol::Make("r"),
                                        BinaryOperator::Div(
                                                BinaryOperator::Pow(
                                                        Symbol::Make("vol"),
                                                        Constant::Make(2.0)
                                                ),
                                                Constant::Make(2.0)
                                        )
                                ),
                                time_to_expiry
                        )
                )
        );

        auto stmt_0 = std::make_shared<Statement>("stmt0", d1);

        
        auto d2 = BinaryOperator::Sub(
                Symbol::Make(stmt_0->Name()),
                BinaryOperator::Mul(
                        Symbol::Make("vol"),
                        time_to_expiry
                )
        );


        auto stmt_1 = std::make_shared<Statement>("stmt1", d2);
        
        auto pv = BinaryOperator::Mul(
                Symbol::Make("K"),
                Exp::Make(
                        BinaryOperator::Mul(
                                BinaryOperator::Sub(
                                        Constant::Make(0.0),
                                        Symbol::Make("r")
                                ),
                                time_to_expiry
                        )
                )
        );
        
        auto stmt_2 = std::make_shared<Statement>("stmt2", pv);

        auto black = BinaryOperator::Sub(
                BinaryOperator::Mul(
                        Phi::Make(stmt_0),
                        Symbol::Make("S")
                ),
                BinaryOperator::Mul(
                        Phi::Make(stmt_1),
                        stmt_2
                )
        );

        auto stmt_3 = std::make_shared<Statement>("stmt3", black);


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

int main(){
        black_scholes();
}
