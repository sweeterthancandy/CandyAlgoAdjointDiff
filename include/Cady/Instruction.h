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
} // end namespace Cady

#endif // INCLUDE_INSTRUCTION_H