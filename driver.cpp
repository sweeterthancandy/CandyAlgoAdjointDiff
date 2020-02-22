#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>

enum OperatorKind{
        OpKind_Unknown,
        OpKind_Statement,
};

#if 0
struct SymbolTable{
private:
        std::unordered_map<std::string, double> m_;
};
#endif

struct Operator : std::enable_shared_from_this<Operator>{
        explicit Operator(OperatorKind kind = OpKind_Unknown):kind_{kind}{}
        virtual ~Operator(){}

        OperatorKind Kind()const{ return kind_; }

        #if 0
        virtual double Eval(SymbolTable const& ST)const=0;
        #endif
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const=0;
        virtual void EmitCode(std::ostream& ss)const=0;
        void Display(std::ostream& out = std::cout)const{
                EmitCode(out);
                out << "\n";
        }


private:
        OperatorKind kind_;
};

struct Constant : Operator{
        Constant(double value)
                :value_(value)
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
private:
        double value_;
};

struct Symbol : Operator{
        Symbol(std::string const& name)
                :name_(name)
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
};

struct BinaryOperator : Operator{
        BinaryOperator(BinaryOperatorKind op, std::shared_ptr<Operator> left, std::shared_ptr<Operator> right)
                :op_(op),
                left_(left),
                right_(right)
        {}
        #if 0
        virtual double Eval(SymbolTable const& ST)const{
                switch(op_)
                {
                case OP_ADD:
                        {
                                return left_->Eval(ST) + right_->Eval(ST);
                        }
                case OP_SUB:
                        {
                                return left_->Eval(ST) - right_->Eval(ST);
                        }
                case OP_MUL:
                        {
                                return left_->Eval(ST) * right_->Eval(ST);
                        }
                case OP_DIV:
                        {
                                return left_->Eval(ST) / right_->Eval(ST);
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
                                        left_->Diff(symbol),
                                        right_->Diff(symbol)
                                );
                        }
                        case OP_SUB:
                        {
                                return Sub(
                                        left_->Diff(symbol),
                                        right_->Diff(symbol)
                                );
                        }
                        case OP_MUL:
                        {
                                return Add(
                                        Mul( left_->Diff(symbol), right_),
                                        Mul( left_, right_->Diff(symbol))
                                );
                        }
                        case OP_DIV:
                        {
                                throw std::domain_error("div todo");
                        }
                }
        }


        virtual void EmitCode(std::ostream& ss)const{
                ss << "(";
                ss << "(";
                left_->EmitCode(ss);
                ss << ")";
                switch(op_){
                case OP_ADD: ss << "+"; break;
                case OP_SUB: ss << "-"; break;
                case OP_MUL: ss << "*"; break;
                case OP_DIV: ss << "/"; break;
                }
                ss << "(";
                right_->EmitCode(ss);
                ss << ")";
                ss << ")";
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

private:
        BinaryOperatorKind op_;
        std::shared_ptr<Operator> left_;
        std::shared_ptr<Operator> right_;
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

        void Emit(std::ostream& ss)const{

                struct VariableInfo{
                        VariableInfo(std::string const& name){
                                name_      = name;
                                diff_name_ = "__diff_" + name;
                        }
                        std::string const& Name()const{ return name_; }
                        std::string const& DiffName()const{ return diff_name_; }
                private:
                        std::string name_;
                        std::string diff_name_;
                };

                std::vector<VariableInfo> deps;
                for( auto const& arg : args_ ){
                        deps.emplace_back(arg);
                }

                ss << "double " << name_ << "(";
                for(size_t idx=0;idx!=deps.size();++idx){
                        ss << "double " << deps[idx].Name() << ", ";
                        ss << " double " << deps[idx].DiffName() << ", ";
                }
                ss << " double* diff";

                ss << ")\n";
                ss << "{\n";

                std::string indent = "    ";

                TemporaryAllocator temp_alloc;

                for(size_t idx=0;idx!=stmts_.size();++idx){
                        // for each statement we need to add two calculations to the
                        // infomation
                        //      statement = expr
                        //      d_statement = D(expr)
                        //      
                        //    

                        //

                        auto const& stmt = stmts_[idx];
                        auto const& expr = stmt->Expr();

                        std::vector<std::string> subs;

                        for( auto const& info : deps ){

                                auto temp_name = temp_alloc.Allocate();

                                // \partial stmt / \partial symbol d symbol
                                auto sub_diff = BinaryOperator::Mul(
                                        expr->Diff( info.Name() ),
                                        Symbol::Make(info.DiffName()));

                                ss << indent << "// \\partial " << stmt->Name() << " / \\partial " << info.Name() << " d " << info.Name() << "\n";
                                ss << indent << "double " << temp_name << " = ";
                                sub_diff->EmitCode(ss);
                                ss << ";\n";

                                subs.push_back(temp_name);
                        }

                        deps.emplace_back(stmt->Name());

                        ss << indent << "double " << deps.back().Name() << " = ";
                        expr->EmitCode(ss);
                        ss << ";\n";


                        ss << indent << "double " << deps.back().DiffName() << " = ";
                        for(size_t idx=0;idx!=subs.size();++idx){
                                if( idx != 0 )
                                        ss << " + ";
                                ss << subs[idx];
                        }
                        ss << ";\n\n\n";

                }

                ss << indent << "*diff = " << deps.back().DiffName() << ";\n";
                ss << indent << "return " << deps.back().Name() << ";\n";
                ss << "}\n";

        }
private:
        std::string name_;
        std::vector<std::string> args_;
        std::vector<std::shared_ptr<Statement> > stmts_;
};


void example_0(){


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
        f.Emit(fstr);
        fstr << R"(
#include <stdio.h>
int main(){
        double a = 2.0;
        double b = 3.0;
        double d_a = 0.0;
        double d_b = 0.0;
        double d_x = 0.05;
        double epsilon = 1e-10;

        for(double x =0.0; x <= 2.0 + d_x /2; x += d_x ){
                double diff = 0.0;
                double y = f(a, 0.0, b, 0.0, x, 1.0, &diff);

                double dummy;
                double lower = f(a, 0.0, b, 0.0, x - epsilon/2, 1.0, &dummy);
                double upper = f(a, 0.0, b, 0.0, x + epsilon/2, 1.0, &dummy);
                double finite_diff = ( upper - lower ) / epsilon;
                
                printf("%f,%f,%f,%f\n", x, y, diff, finite_diff);


        }

}
)";

}


int main(){
        example_0();
}
