#ifndef INCLUDE_CADY_H
#define INCLUDE_CADY_H

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <cmath>

#include <boost/optional.hpp>

namespace std{
        template< class T, class U > 
        std::shared_ptr<T> reinterpret_pointer_cast( const std::shared_ptr<U>& r ) noexcept
        {
                    auto p = reinterpret_cast<typename std::shared_ptr<T>::element_type*>(r.get());
                        return std::shared_ptr<T>(r, p);
        }
} // end namespace std






/* 
                                Ideas
                                =====
        1) We could do something where look for common subtrees with a metric
           induced by assigning weights to nodes, IE div is 50, plus is 1, 
           Phi is 1000. Then we could count all subtrees
                        root = external-root
                        for(;;){
                                compute a statistic of counts of common subtrees
                                take the largest neaive saving, with is ( count -1 ) * metric,
                                if the subtree is not above a threshold
                                        break
                                and replace all instanace
                        }
*/

namespace Cady{

enum OperatorKind{
        OPKind_UnaryOperator,
        OPKind_BinaryOperator,
        OPKind_Constant,
        OPKind_ExogenousSymbol,
        OPKind_EndgenousSymbol,
        OPKind_Other,
};


struct EndgenousSymbol;
using  EndgenousSymbolSet = std::unordered_set<std::shared_ptr<EndgenousSymbol > >;
struct Operator : std::enable_shared_from_this<Operator>{

        explicit Operator(std::string const& name, OperatorKind kind = OPKind_Other)
                : name_{name}
                , kind_{kind}
        {}
        virtual ~Operator(){}

        OperatorKind Kind()const{ return kind_; }




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
        void Rebind(size_t idx, std::shared_ptr<Operator> const& ptr){
                if( Arity() < idx ){
                        throw std::domain_error("getting child that doesn't exist");
                }
                children_.at(idx) = ptr;
        }
        auto&       Children()     { return children_; }
        auto const& Children()const{ return children_; }

        bool IsTerminal()const{ return Arity() == 0; }
        bool IsNonTerminal()const{ return Arity() > 0; }

        std::string const& Name()const{ return name_; }
        std::string NameWithHidden()const{
                std::stringstream ss;
                ss << name_ << "<" << this << ">" << "{";
                auto hidden = HiddenArguments();
                for(size_t idx=0;idx!=hidden.size();++idx){
                        if( idx != 0 )
                                ss << ", ";
                        ss << hidden[idx];
                }
                ss << "}";
                return ss.str();
        }

        virtual std::vector<std::string> HiddenArguments()const{ return {}; }

        inline void Display(std::ostream& ostr = std::cout);

        EndgenousSymbolSet EndgenousDependencies(){
                EndgenousSymbolSet mem;
                EndgenousDependenciesCollect(mem);
                return mem;
        }
        virtual void EndgenousDependenciesCollect(EndgenousSymbolSet& mem){
                std::vector<std::shared_ptr<Operator > > stack{shared_from_this()};
                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        #if 0
                        std::cout << std::string(stack.size()*2, ' ') << head->NameWithHidden() << "\n";
                        #endif
                        for(auto& ptr : head->children_){
                                if( ptr->Kind() == OPKind_EndgenousSymbol){
                                        mem.insert(std::reinterpret_pointer_cast<EndgenousSymbol>(ptr));
                                } else {
                                        stack.push_back(ptr);
                                }
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

        OperatorKind kind_;

};


struct Constant : Operator{
        Constant(double value)
                :Operator{"Constant", OPKind_Constant}
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
        double Value()const{ return value_; }
private:
        double value_;
};

struct ExogenousSymbol : Operator{
        ExogenousSymbol(std::string const& name)
                :Operator{"ExogenousSymbol", OPKind_ExogenousSymbol}
                ,name_(name)
        {}
        virtual std::vector<std::string> HiddenArguments()const{ return {name_}; }
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
        
        static std::shared_ptr<ExogenousSymbol> Make(std::string const& symbol){
                return std::make_shared<ExogenousSymbol>(symbol);
        }
private:
        std::string name_;
};
/*
        I want a mechinism for follow statements.
        there are two cases, a symbol which represents
        something exogenous, and the case we we are
        splitting a 

 */
struct EndgenousSymbol : Operator{
        EndgenousSymbol(std::string const& name,
                        std::shared_ptr<Operator> const& expr)
                :Operator{"EndgenousSymbol", OPKind_EndgenousSymbol}
                ,name_{name}
        {
                Push(expr);
        }
        virtual std::vector<std::string> HiddenArguments()const{ return {name_, "<expr>"}; }
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
        
        static std::shared_ptr<EndgenousSymbol> Make(std::string const& symbol, std::shared_ptr<Operator> const& expr){
                return std::make_shared<EndgenousSymbol>(symbol, expr);
        }

        std::shared_ptr<Operator> Expr()const{ return At(0); }
        std::shared_ptr<Operator> as_operator_()const{ return At(0); }
        
        #if 0
        virtual void EndgenousDependenciesCollect(EndgenousSymbolSet& mem)const{
                mem.insert(std::reinterpret_pointer_cast<EndgenousSymbol const>(shared_from_this()));
        }
        #endif
private:
        std::string name_;
};
        
void Operator::Display(std::ostream& ostr){

        #if 0
        std::cout << "this->Name() => " << this->Name() << "\n"; // __CandyPrint__(cxx-print-scalar,this->Name())
        for(auto const& ptr : EndgenousDependencies() ){
                std::cout << "    : " << ptr->Name() << "\n";
        }
        #endif

        using ptr_t = std::shared_ptr<Operator const>;
        std::vector<std::vector<ptr_t> > stack{{shared_from_this()}};

        auto indent = [&](int extra = 0){
                return std::string((stack.size()+extra)*2,' ');
        };

        for(size_t ttl=1000;stack.size() && ttl;--ttl){

                #if 0
                std::cout << "{";
                for(size_t j=0;j!=stack.size();++j){
                        if( j!=0)
                                std::cout << ",";
                        std::cout << stack[j].size();
                }
                std::cout << "}\n";
                #endif



                auto& head = stack.back();

                if( head.empty() ){
                        stack.pop_back();
                        if( stack.size() == 0 )
                                break;
                        ostr << indent() << "}\n";
                        continue;
                }

                auto ptr = head.back();
                head.pop_back();



                auto hidden = ptr->HiddenArguments();
                if( ptr->IsTerminal() ){
                        #if 1
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
                        #endif
                } else {
                        ostr << indent() << ptr->Name() << "{\n";
                        for(auto const& s : hidden ){
                                ostr << indent(1) << s << "\n";
                        }
                        auto children = ptr->Children();
                        #if 0
                        std::cout << "stack.size() => " << stack.size() << "\n"; // __CandyPrint__(cxx-print-scalar,stack.size())
                        stack.emplace_back();
                        for(auto iter = children.rbegin(), end = children.rend();iter!=end;++iter){
                                stack.back().push_back(*iter);
                        }
                        std::cout << "stack.size() => " << stack.size() << "\n"; // __CandyPrint__(cxx-print-scalar,stack.size())
                        #endif
                        stack.emplace_back(children.rbegin(), children.rend());
                }
        }
}

enum UnaryOperatorKind{
        UOP_USUB,
};

struct UnaryOperator : Operator{
        UnaryOperator(UnaryOperatorKind op, std::shared_ptr<Operator> arg)
                :Operator{"UnaryOperator", OPKind_UnaryOperator}
                ,op_(op)
        {
                Push(arg);
        }

        UnaryOperatorKind OpKind()const{ return op_; }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override
        {
                return UnaryMinus(At(0)->Diff(symbol));
        }

        static std::shared_ptr<Operator> UnaryMinus(std::shared_ptr<Operator> const& arg){
                return std::make_shared<UnaryOperator>(UOP_USUB, arg);
        }




        virtual void EmitCode(std::ostream& ss)const override{
                ss << "(-(";
                At(0)->EmitCode(ss);
                ss << "))";
        }


private:
        UnaryOperatorKind op_;
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
                :Operator{"BinaryOperator", OPKind_BinaryOperator}
                ,op_(op)
        {
                Push(left);
                Push(right);
        }

        BinaryOperatorKind OpKind()const{ return op_; }
        virtual std::vector<std::string> HiddenArguments()const override{
                switch(op_){
                case OP_ADD: return {"ADD"};
                case OP_SUB: return {"SUB"};
                case OP_MUL: return {"MUL"};
                case OP_DIV: return {"DIV"};
                case OP_POW: return {"POW"};
                }
                return {"unknown_"};
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
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override
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


        virtual void EmitCode(std::ostream& ss)const override{
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
};

struct Exp : Operator{
        Exp(std::shared_ptr<Operator> arg)
                :Operator{"Exp"}
        {
                Push(arg);
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                return BinaryOperator::Mul(
                        std::make_shared<Exp>(At(0)),
                        At(0)->Diff(symbol));
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << "std::exp(";
                At(0)->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Exp> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Exp>(arg);
        }
};

struct Log : Operator{
        Log(std::shared_ptr<Operator> arg)
                :Operator{"Log"}
        {
                Push(arg);
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const{
                return BinaryOperator::Div(
                        At(0)->Diff(symbol),
                        At(0));
        }
        virtual void EmitCode(std::ostream& ss)const{
                ss << "std::log(";
                At(0)->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Log> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Log>(arg);
        }
};


// normal distribution CFS
struct Phi : Operator{
        Phi(std::shared_ptr<Operator> arg)
                :Operator{"Phi"}
        {
                Push(arg);
        }
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
                                                                        At(0),
                                                                        Constant::Make(2.0)
                                                                )
                                                        )
                                                )
                                        ),
                                        Constant::Make(2.506628274631000502415765284811045253006986740609938316629)
                                ),
                                At(0)->Diff(symbol)
                        );

        }
        virtual void EmitCode(std::ostream& ss)const{
                // std::erfc(-x/std::sqrt(2))/2
                ss << "std::erfc(-(";
                At(0)->EmitCode(ss);
                ss << ")/std::sqrt(2))/2";
        }

        static std::shared_ptr<Operator> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Phi>(arg);
        }
};

#if 0
struct Statement : EndgenousSymbol{
        Statement(std::string const& name, std::shared_ptr<Operator> expr)
                :ExogenousSymbol(name),
                expr_(expr)
        {}
        std::shared_ptr<Operator> Expr()const{ return expr_; }
        std::shared_ptr<Operator> as_operator_()const{ return expr_; }
private:
        std::string name_;
        std::shared_ptr<Operator> expr_;
};
#endif

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
        auto AddStatement(std::shared_ptr<EndgenousSymbol> stmt){
                stmts_.push_back(stmt);
                return stmt;
        }
        auto const& Arguments()const{ return args_; }
        auto const& Statements()const{ return stmts_; }
        std::string const& Name()const{ return name_; }

private:
        std::string name_;
        std::vector<std::string> args_;
        std::vector<std::shared_ptr<EndgenousSymbol> > stmts_;
};

struct ConstantDescription{
        ConstantDescription(std::shared_ptr<Operator> root){
                if( root->Kind() != OPKind_Constant )
                        return;
                auto constant = static_cast<Constant*>(root.get());
                opt_value_ = constant->Value();
        }
        bool IsZero()const{ return opt_value_ && (*opt_value_ == 0.0 || *opt_value_ == -0.0); }
        bool IsOne()const{ return opt_value_ && *opt_value_ == 1.0; }
        bool IsConstantValue()const{ return !! opt_value_; }
        double ValueOrThrow()const{
                if( ! opt_value_ )
                        throw std::domain_error("have no value");
                return opt_value_.get();
        }
private:
        boost::optional<double> opt_value_;
};

struct FoldZero{

        std::shared_ptr<Operator> Fold(std::shared_ptr<Operator> root){
                if( root->Kind() == OPKind_BinaryOperator ){
                        auto bin_op = static_cast<BinaryOperator*>(root.get());

                        auto left_folded = this->Fold(bin_op->At(0));
                        auto right_folded = this->Fold(bin_op->At(1));

                        auto left_desc  = ConstantDescription{left_folded};
                        auto right_desc = ConstantDescription{right_folded};


                        switch(bin_op->OpKind())
                        {
                                case OP_ADD:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                double sum = 0.0;
                                                sum += left_desc.ValueOrThrow();
                                                sum += right_desc.ValueOrThrow();
                                                return Constant::Make(sum);
                                        }
                                        if( left_desc.IsZero() )
                                                return right_folded;
                                        if( right_desc.IsZero() )
                                                return left_folded;


                                        break;
                                }
                                case OP_SUB:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                double sum = 0.0;
                                                sum += left_desc.ValueOrThrow();
                                                sum -= right_desc.ValueOrThrow();
                                                return Constant::Make(sum);
                                        }

                                        if( right_desc.IsZero())
                                                return left_folded;

                                        if( left_desc.IsZero() ){
                                                return UnaryOperator::UnaryMinus(right_folded);
                                        }
                                        break;
                                }
                                case OP_MUL:
                                {
                                        if( left_desc.IsZero() || right_desc.IsZero() ){
                                                return Constant::Make(0.0);
                                        }
                                        if( left_desc.IsOne() )
                                                return right_folded;
                                        if( right_desc.IsOne() )
                                                return left_folded;
                                        break;
                                }
                                case OP_DIV:
                                {
                                        if( ! left_desc.IsZero() ){
                                                if( right_desc.IsZero() ){
                                                        throw std::domain_error("have divide by zero");
                                                }
                                        } else {
                                                return Constant::Make(0.0);
                                        }
                                        break;
                                }
                                case OP_POW:
                                {
                                        if( left_desc.IsConstantValue() && 
                                            right_desc.IsConstantValue() ){
                                                return Constant::Make(
                                                        std::pow(
                                                                left_desc.ValueOrThrow()
                                                                ,right_desc.ValueOrThrow()
                                                        )
                                                );
                                        }

                                        if( left_desc.IsZero() )
                                                return Constant::Make(0.0);

                                        if( right_desc.IsOne() ){
                                                return left_folded;
                                        }
                                        if( right_desc.IsZero() ){
                                                return Constant::Make(1.0);
                                        }
                                        break;
                                }
                        }
                        return std::make_shared<BinaryOperator>(
                                bin_op->OpKind(),
                                left_folded,
                                right_folded);
                }
                
                if( root->Kind() == OPKind_UnaryOperator ){
                        auto unary_op = static_cast<UnaryOperator*>(root.get());

                        auto folded_arg = this->Fold(unary_op->At(0));

                        auto arg_desc  = ConstantDescription{folded_arg};

                        if( arg_desc.IsZero() ){
                                return Constant::Make(0.0);
                        }

                        return UnaryOperator::UnaryMinus(folded_arg);
                }

                if( root->IsNonTerminal() ){
                        for(size_t idx=0;idx!=root->Arity();++idx){
                                auto folded = this->Fold(root->At(idx));
                                root->Rebind(idx, folded);
                        }
                }
                return root;
        }
};


struct StringCodeGenerator{
        void Emit(std::ostream& ss, Function const& f)const{

                FoldZero folder;

                // we have a vector [ x1, x2, ... ] which are the function 
                // parameters. 

                struct VariableInfo{
                        VariableInfo(std::string const& name)
                                : name_{name}
                        {}
                        std::string const& Name()const{ return name_; }
                        boost::optional<std::shared_ptr<Operator> > GetDiffLexical(std::string const& symbol)const{
                                auto iter = diff_map_.find(symbol);
                                if( iter == diff_map_.end() )
                                        return {};
                                return iter->second;
                        }
                        void MapDiff(std::string const& symbol,
                                    std::shared_ptr<Operator> const& value){
                                diff_map_[symbol] = value;
                        }
                private:
                        std::string name_;
                        std::unordered_map<std::string, std::shared_ptr<Operator> > diff_map_;
                };


                auto to_diff = f.Arguments();


                std::vector<std::shared_ptr<VariableInfo> > deps;
                for( auto const& arg : f.Arguments() ){
                        auto ptr = std::make_shared<VariableInfo>(arg);
                        for( auto const& inner_arg : to_diff ){
                                if( arg == inner_arg ){
                                        ptr->MapDiff(inner_arg, Constant::Make(1.0));
                                } else {
                                        ptr->MapDiff(inner_arg, Constant::Make(0.0));
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
                        
                        #if 0
                        ss << indent << "/* expr\n";
                        expr->Display(ss);
                        ss << indent << "*/\n";
                        #endif
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
                                                *info->GetDiffLexical(d_symbol));

                                        sub_diff = folder.Fold(sub_diff);

                                        if( sub_diff->Kind() == OPKind_Constant  ){
                                                auto constant = reinterpret_cast<Constant*>(sub_diff.get());
                                                if( constant->Value() == 0.0 )
                                                        continue;
                                        }


                                        #if 0
                                        ss << indent << "// \\partial " << stmt->Name() << " / \\partial " << info->Name() << " d " << info->Name() << "\n";
                                        #endif
                                        #if 0
                                        ss << indent << "/* expr\n";
                                        sub_diff->Display(ss);
                                        ss << indent << "*/\n";
                                        #endif
                                        ss << indent << "double " << temp_name << " = ";
                                        sub_diff->EmitCode(ss);
                                        ss << ";\n";

                                        subs.push_back(temp_name);
                                }


                                std::string token = "__diff_" + stmt->Name() + "_" + d_symbol;
                                stmt_dep->MapDiff( d_symbol, ExogenousSymbol::Make(token));


                                ss << indent << "double " << token << " = ";
                                if( subs.size() ){
                                        for(size_t idx=0;idx!=subs.size();++idx){
                                                if( idx != 0 )
                                                        ss << " + ";
                                                ss << subs[idx];
                                        }
                                } else {
                                        ss << "0.0";
                                }
                                ss << ";\n";
                        }
                        ss << "\n\n\n";
                        deps.emplace_back(stmt_dep);

                }
                        
                for( auto const& d_symbol : to_diff ){
                        ss << indent << "*d_" + d_symbol << " = " << reinterpret_cast<ExogenousSymbol*>(deps.back()->GetDiffLexical(d_symbol).get().get())->Name() << ";\n";
                }

                ss << indent << "return " << deps.back()->Name() << ";\n";
                ss << "}\n";

        }
};



} // end namespace Cady

#endif // INCLUDE_CANDY_CADY_H
