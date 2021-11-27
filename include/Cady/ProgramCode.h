#ifndef CADY_PROGRAM_CODE_H
#define CADY_PROGRAM_CODE_H

#include "Cady/SourceCodeManager.h"

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
            StatementList(std::vector<std::shared_ptr< Statement> > const& args) :
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
                : op_{ op }, rvalue_{
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
                , if_false_{ if_false }
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
                : function_name_{ function_name }
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
                : name_{ name }, args_{
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
            auto const& Statements()const { return stmts_; }
            auto& Statements() { return stmts_; }
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
            void EmitCode(std::ostream& ostr, std::shared_ptr<SourceCodeManager> const& mgr, std::shared_ptr< Function> const& f)const
            {
                ostr << "auto " << f->Name() << "(";
                auto const& args = f->Args();
                for (size_t idx = 0; idx != args.size(); ++idx)
                {
                    ostr << (idx == 0 ? "" : ", ") << "const double " << args[idx];
                }
                ostr << ")\n";
                ostr << "{\n";
                EmitCodeForStatement(ostr, mgr, f->Statements(), 1);
                ostr << "}\n";
            }
            void EmitCodeForStatement(std::ostream& ostr, std::shared_ptr<SourceCodeManager> const& mgr, std::shared_ptr<Statement> const& stmt, size_t indent)const
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
                    EmitCodeForStatement(ostr, mgr, if_stmt->if_true_, indent + 1);
                    do_indent();
                    std::cout << "} else {\n";
                    EmitCodeForStatement(ostr, mgr, if_stmt->if_false_, indent + 1);
                    do_indent();
                    std::cout << "}\n";
                }
                else if (auto stmts = std::dynamic_pointer_cast<StatementList>(stmt))
                {
                    for (auto const& x : *stmts)
                    {
                        EmitCodeForStatement(ostr, mgr, x, indent);
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
                    if (mgr->GetCC() == CC_ReturnScalar)
                    {
                        if (call_stmt->result_list_.size() != 1)
                        {
                            throw std::domain_error("calling convention demands a single result");
                        }
                        auto const& lvalue = call_stmt->result_list_[0];
                        do_indent();
                        ostr << "const double " << lvalue->ToString() << " = " << call_stmt->function_name_ << "(";
                        for (size_t idx = 0; idx != call_stmt->arg_list_.size(); ++idx)
                        {
                            ostr << (idx == 0 ? "" : ", ") << call_stmt->arg_list_[idx]->ToString();
                        }
                        ostr << ");\n";
                        
                    }
                    else
                    {
                        do_indent();
                        std::string call_result_token = "__call_result_" + std::to_string((std::size_t)call_stmt.get());
                        ostr << "auto " << call_result_token << " = " << call_stmt->function_name_ << "(";
                        for (size_t idx = 0; idx != call_stmt->arg_list_.size(); ++idx)
                        {
                            ostr << (idx == 0 ? "" : ", ") << call_stmt->arg_list_[idx]->ToString();
                        }
                        ostr << ");\n";
                        for (size_t idx = 0; idx != call_stmt->result_list_.size(); ++idx)
                        {
                            auto const& lvalue = call_stmt->result_list_[idx];
                            do_indent();
                            ostr << "const double " << lvalue->ToString() << " = " << call_result_token << "[" << idx << "];\n";
                        }
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

#endif // CADY_PROGRAM_CODE_H