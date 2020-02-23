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
#include <iomanip>

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
using  EndgenousSymbolVec = std::vector<std::shared_ptr<EndgenousSymbol > >;

struct Operator;

using OperatorVector = std::vector<std::shared_ptr<Operator> >;

struct SymbolTable{
        SymbolTable& operator()(std::string const& sym, double value){
                m_[sym] = value;
                return *this;
        }
        double operator[](std::string const& sym)const{
                auto iter = m_.find(sym);
                if( iter == m_.end()){
                        std::stringstream ss;
                        ss << "symbol " << sym << " does not exist";
                        throw std::domain_error(ss.str());
                }
                return iter->second;
        }
private:
        std::unordered_map<std::string, double> m_;
};

struct Operator : std::enable_shared_from_this<Operator>{

        explicit Operator(std::string const& name, OperatorKind kind = OPKind_Other)
                : name_{name}
                , kind_{kind}
        {}
        virtual ~Operator(){}

        OperatorKind Kind()const{ return kind_; }

        inline void MutateToEndgenous(std::string const& name);


        virtual std::shared_ptr<Operator> Clone()const=0;

        
        EndgenousSymbolVec DepthFirstAnySymbolicDependency(){
                EndgenousSymbolVec result;
                CollectDepthFirstAnySymbolicDependency(result);
                return result;
        }
        void CollectDepthFirstAnySymbolicDependency(EndgenousSymbolVec& mem){
                std::vector<std::shared_ptr<Operator > > stack{shared_from_this()};
                for(;stack.size();){
                        auto head = stack.back();
                        stack.pop_back();
                        for(auto& ptr : head->children_){
                                if( ptr->Kind() == OPKind_EndgenousSymbol ){
                                        ptr->CollectDepthFirstAnySymbolicDependency(mem);
                                        mem.push_back(std::reinterpret_pointer_cast<EndgenousSymbol>(ptr));
                                } else {
                                        stack.push_back(ptr);
                                }
                        }
                }
        }

        struct EvalChecker{
                void Push(std::shared_ptr<Operator const> const& ptr){
                        if( depth_.count(ptr) > 0 ){
                                for(size_t idx=0;idx!=seq_.size();++idx){
                                        std::string token = ( ptr == seq_[idx] ? "->" : "  " );
                                        std::cout << token << "[" << std::setw(2) << idx << "] : " << seq_[idx]->NameInvariantOfChildren() << "\n";
                                }
                                throw std::domain_error("recursive eval");
                        }
                        depth_.insert(ptr);
                        seq_.push_back(ptr);
                }
                void Pop(){
                        depth_.erase(seq_.back());
                        seq_.pop_back();
                }

        private:
                std::unordered_set<std::shared_ptr<Operator const> > depth_;
                std::vector<std::shared_ptr<Operator const> > seq_;
        };

        struct EvalCheckerDevice{
                EvalCheckerDevice(EvalChecker& checker, std::shared_ptr<Operator const> ptr)
                        : checker_{nullptr}
                {
                        checker.Push(ptr);
                        checker_ = &checker;
                }
                ~EvalCheckerDevice(){
                        if( checker_ )
                                checker_->Pop();
                }
        private:
                EvalChecker* checker_;
        };


        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const=0;
        double Eval(SymbolTable const& ST)const{
                EvalChecker eval_checker;
                return EvalImpl(ST, eval_checker);
        }

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
        std::string NameInvariantOfChildren()const{
                std::stringstream ss;
                ss << name_ << "{";
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

        std::string endo_name_;
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
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return value_;
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
                return Constant::Make(0.0);
        }
        virtual void EmitCode(std::ostream& ss)const override{
                ss << value_;
        }

        static std::shared_ptr<Operator> Make(double value){
                return std::make_shared<Constant>(value);
        }
        virtual std::vector<std::string> HiddenArguments()const override{ return { std::to_string(value_) }; }
        double Value()const{ return value_; }
        virtual std::shared_ptr<Operator> Clone()const override{ return Make(value_); }
private:
        double value_;
};

struct ExogenousSymbol : Operator{
        ExogenousSymbol(std::string const& name)
                :Operator{"ExogenousSymbol", OPKind_ExogenousSymbol}
                ,name_(name)
        {}
        virtual std::vector<std::string> HiddenArguments()const override{ return {name_}; }
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return ST[name_];
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
                if( symbol == name_ ){
                        return Constant::Make(1.0);
                }
                return Constant::Make(0.0);
        }
        virtual void EmitCode(std::ostream& ss)const override{
                ss << name_;
        }
        std::string const& Name()const{ return name_; }
        
        static std::shared_ptr<ExogenousSymbol> Make(std::string const& symbol){
                return std::make_shared<ExogenousSymbol>(symbol);
        }
        virtual std::shared_ptr<Operator> Clone()const override{ return Make(name_); }
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
        {
                Push(expr);
                endo_name_ = name;
        }
        virtual std::vector<std::string> HiddenArguments()const override{ return {endo_name_, "<expr>"}; }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
                if( symbol == endo_name_ ){
                        return Constant::Make(1.0);
                }
                return Constant::Make(0.0);
        }
        virtual void EmitCode(std::ostream& ss)const override{
                ss << endo_name_;
        }
        std::string const& Name()const{ return endo_name_; }
        
        static std::shared_ptr<EndgenousSymbol> Make(std::string const& symbol, std::shared_ptr<Operator> const& expr){
                return std::make_shared<EndgenousSymbol>(symbol, expr);
        }

        std::shared_ptr<Operator> Expr()const{ return At(0); }
        std::shared_ptr<Operator> as_operator_()const{ return At(0); }
        
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                EvalCheckerDevice device(checker, shared_from_this());
                return At(0)->EvalImpl(ST, checker);
        }
        
        virtual std::shared_ptr<Operator> Clone()const override{ return Make(endo_name_, At(0)); }
#if 0
private:
        std::string name_;
#endif
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
        virtual std::shared_ptr<Operator> Clone()const override{
                return std::make_shared<UnaryOperator>(
                        op_,
                        At(0)
                );
        }
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return -At(0)->EvalImpl(ST, checker);
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
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                switch(op_)
                {
                case OP_ADD:
                        {
                                return At(0)->EvalImpl(ST, checker) + At(1)->EvalImpl(ST, checker);
                        }
                case OP_SUB:
                        {
                                return At(0)->EvalImpl(ST, checker) - At(1)->EvalImpl(ST, checker);
                        }
                case OP_MUL:
                        {
                                return At(0)->EvalImpl(ST, checker) * At(1)->EvalImpl(ST, checker);
                        }
                case OP_DIV:
                        {
                                return At(0)->EvalImpl(ST, checker) / At(1)->EvalImpl(ST, checker);
                        }
                case OP_POW:
                        {
                                return std::pow(At(0)->EvalImpl(ST, checker),At(1)->EvalImpl(ST, checker));
                        }
                }
        }
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

        virtual std::shared_ptr<Operator> Clone()const override{
                return std::make_shared<BinaryOperator>(
                        op_,
                        At(0),
                        At(1)
                );
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
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
                return BinaryOperator::Mul(
                        std::make_shared<Exp>(At(0)),
                        At(0)->Diff(symbol));
        }
        virtual void EmitCode(std::ostream& ss)const override{
                ss << "std::exp(";
                At(0)->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Exp> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Exp>(arg);
        }
        virtual std::shared_ptr<Operator> Clone()const override{
                return Make(At(0));
        }
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return std::exp(At(0)->EvalImpl(ST, checker));
        }
};

struct Log : Operator{
        Log(std::shared_ptr<Operator> arg)
                :Operator{"Log"}
        {
                Push(arg);
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
                return BinaryOperator::Div(
                        At(0)->Diff(symbol),
                        At(0));
        }
        virtual void EmitCode(std::ostream& ss)const override{
                ss << "std::log(";
                At(0)->EmitCode(ss);
                ss << ")";
        }
        static std::shared_ptr<Log> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Log>(arg);
        }
        virtual std::shared_ptr<Operator> Clone()const override{
                return Make(At(0));
        }
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return std::log(At(0)->EvalImpl(ST, checker));
        }
};


// normal distribution CFS
struct Phi : Operator{
        Phi(std::shared_ptr<Operator> arg)
                :Operator{"Phi"}
        {
                Push(arg);
        }
        virtual std::shared_ptr<Operator> Diff(std::string const& symbol)const override{
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
        virtual void EmitCode(std::ostream& ss)const override{
                // std::erfc(-x/std::sqrt(2))/2
                ss << "std::erfc(-(";
                At(0)->EmitCode(ss);
                ss << ")/std::sqrt(2))/2";
        }
        virtual double EvalImpl(SymbolTable const& ST, EvalChecker& checker)const override{
                return std::erfc(-At(0)->EvalImpl(ST, checker)/std::sqrt(2))/2;
        }

        static std::shared_ptr<Operator> Make(std::shared_ptr<Operator> const& arg){
                return std::make_shared<Phi>(arg);
        }
        virtual std::shared_ptr<Operator> Clone()const override{
                return Make(At(0));
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


inline void Operator::MutateToEndgenous(std::string const& name){
        auto clone = this->Clone();
        this->~Operator();
        new(this)EndgenousSymbol(name, clone);
}




} // end namespace Cady

#endif // INCLUDE_CANDY_CADY_H
