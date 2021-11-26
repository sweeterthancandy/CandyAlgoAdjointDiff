#ifndef INCLUDE_INSTRUCTION_H
#define INCLUDE_INSTRUCTION_H

#include "Cady.h"

namespace Cady {


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
        std::string const& LValueName()const { return name_; }
    private:
        std::string name_;
        std::shared_ptr<Operator> op_;
    };



    struct InstructionDeclareMatrix : Instruction {
        InstructionDeclareMatrix(std::string const& name, std::shared_ptr<SymbolicMatrix> const& matrix)
            :Instruction{ Instr_VarDecl }
            , name_(name)
            , matrix_(matrix)
        {}
        virtual void EmitCode(std::ostream& out)const {
            size_t rows = matrix_->get_impl().size();
            size_t cols = matrix_->get_impl()[0].size();
            out << "    const Eigen::Matrix<double, " << rows << "," << cols << ">" << name_ << " {\n";
            for (size_t i = 0; i != rows; ++i)
            {
                out << (i == 0 ? "" : ", ") << "{";
                for (size_t j = 0; j != cols; ++j)
                {
                    out << (j == 0 ? "" : ", ");
                    matrix_->get_impl().at(i).at(j)->EmitCode(out);
                }
                out << "}\n";
            }
            out << "};\n";
        }
        std::string const& LValueName()const { return name_; }
        std::shared_ptr<SymbolicMatrix> const& Matrix()const {
            return matrix_;
        }
    private:
        std::string name_;
        std::shared_ptr<SymbolicMatrix> matrix_;
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
        std::string const& VarName()const { return name_;  }
    private:
        std::string name_;
    };

    struct IfBlock;
    struct CallBlock;

    struct ControlBlockVisitor
    {
        virtual ~ControlBlockVisitor() = default;
        virtual void AcceptInstruction(const std::shared_ptr<const Instruction>& instr)= 0;
        virtual void AcceptIf(const std::shared_ptr<const IfBlock>& if_block)= 0;
        virtual void AcceptCall(const std::shared_ptr<const CallBlock>& call_block)= 0;
    };

   


    struct ControlBlock : std::enable_shared_from_this<ControlBlock>
    {
        virtual ~ControlBlock() = default;
        virtual void EmitCode(std::ostream& out)const = 0;
        virtual void Accept(ControlBlockVisitor& V)const = 0;
    };
    

    // instruction block has not branching
    struct InstructionBlock : ControlBlock, std::vector<std::shared_ptr<Instruction> > {
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
        virtual void Accept(ControlBlockVisitor& V)const override
        {
            for (auto const& ptr : *this) {
                V.AcceptInstruction(ptr);
            }
        }
    };


    struct IfBlock : ControlBlock
    {
        IfBlock(
            std::string const& cond,
            std::shared_ptr< ControlBlock> const& if_true,
            std::shared_ptr< ControlBlock> const& if_false)
            : cond_{ cond }, if_true_{ if_true }, if_false_{ if_false }
        {}
        std::string const& ConditionVariable()const { return cond_;  }
        std::shared_ptr<ControlBlock> const& IfTrue()const { return if_true_;  }
        std::shared_ptr<ControlBlock> const& IfFalse()const { return if_false_; }
        void EmitCode(std::ostream& out)const
        {
            out << "    if( !! " << cond_ << " )\n";
            out << "    {\n";
            if_true_->EmitCode(out);
            out << "    }\n";
            out << "    else\n";
            out << "    {\n";
            if_false_->EmitCode(out);
            out << "    }\n";
        }
        virtual void Accept(ControlBlockVisitor& V)const override
        {
            V.AcceptIf(std::static_pointer_cast<const IfBlock>(shared_from_this()));
        }
    private:
        std::string cond_;
        std::shared_ptr<ControlBlock> if_true_;
        std::shared_ptr<ControlBlock> if_false_;
    };

    struct CallBlock : ControlBlock
    {
        explicit CallBlock(
            std::string const& return_name,
            std::vector < std::string> const& args) :args_{ args } {}
        void EmitCode(std::ostream& out)const
        {
            out << "double " << return_name_ << " = f(";
            for (size_t idx = 0; idx != args_.size(); ++idx)
            {
                out << (idx == 0 ? "" : ", ") << args_[idx];
            }
            out << ")";
        }
        virtual void Accept(ControlBlockVisitor& V)const override
        {
            V.AcceptCall(std::static_pointer_cast<const CallBlock>(shared_from_this()));
        }
    private:
        std::string return_name_;
        std::vector<std::string> args_;
    };

    struct Module : ControlBlock, std::vector<std::shared_ptr<ControlBlock> >
    {
        void EmitCode(std::ostream& out)const
        {
            for (auto const& x : *this)
            {
                x->EmitCode(out);
            }
        }
        virtual void Accept(ControlBlockVisitor& V)const override
        {
            for (auto const& x : *this)
            {
                x->Accept(V);
            }
        }
    };


    enum FunctionArgumentKind
    {
        FAK_Double,
        FAK_OptDoublePtr,
    };

    struct FunctionArgument
    {
        FunctionArgument(
            FunctionArgumentKind kind,
            std::string const& name) :
            kind_{ kind }, name_{ name }
        {}
        FunctionArgumentKind Kind()const { return kind_; }
        std::string const& Name()const { return name_; }
    private:
        FunctionArgumentKind kind_;
        std::string name_;
    };

    // this is a WIP, at some point want to add branching
    struct Function
    {
        explicit Function(
            std::shared_ptr<Module> const& modulee)
            :module_{ modulee } {}
        explicit Function(
            std::shared_ptr<InstructionBlock> const& IB)
        {
            module_ = std::make_shared<Module>();
            module_->push_back(IB);
        }
        void AddArg(std::shared_ptr < FunctionArgument> const& arg)
        {
            args_.push_back(arg);
        }
        auto args_begin()const { return std::begin(args_); }
        auto args_end()const { return std::end(args_); }
        void SetFunctionName(std::string const& func_name)
        {
            func_name_ = func_name;
        }
        std::string const& FunctionName()const {
            return func_name_;
        }
        auto const& GetModule()const { return module_; }
    private:
        std::shared_ptr<Module> module_;
        std::string func_name_{ "unnamed_function" };
        std::vector<std::shared_ptr< FunctionArgument> > args_;
    };


} // end namespace Cady

#endif // INCLUDE_INSTRUCTION_H