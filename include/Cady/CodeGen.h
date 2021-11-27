#ifndef INCLUDE_CADY_CODEGEN_H
#define INCLUDE_CADY_CODEGEN_H

#include "Cady.h"
#include "Transform.h"

namespace Cady{
namespace CodeGen{

    std::shared_ptr<Module> BuildRecursiveEx(std::shared_ptr<Operator> const& head)
    {
        struct CtrlFinder
        {
            void operator()(std::shared_ptr<Operator> const& op)
            {
                if (auto ptr = std::dynamic_pointer_cast<If>(op))
                {
                    ctrl_stmts_.push_back(ptr);
                }
            }

            std::vector<std::shared_ptr<Operator> > ctrl_stmts_;
        };

        auto ctrl_finder = std::make_shared< CtrlFinder>();
        head->VisitTopDown(*ctrl_finder);

        if (ctrl_finder->ctrl_stmts_.size() > 0)
        {
            auto untyped_ctrl_stmt = ctrl_finder->ctrl_stmts_[0];
            if (auto if_stmt = std::dynamic_pointer_cast<If>(untyped_ctrl_stmt))
            {
                auto if_expr = if_stmt;

                // I expect this to be a block of variable declerations,
                // the last decleration will be the conditional variable
                auto cond = BuildRecursiveEx(if_expr->Cond());
                auto cond_module = std::dynamic_pointer_cast<Module>(cond);
                auto cond_ib = std::dynamic_pointer_cast<InstructionBlock>(cond_module->back());

                auto cond_var_name = [&]()->std::string
                {
                    if (auto last_instr = std::dynamic_pointer_cast<InstructionReturn>(cond_ib->back()))
                    {
                        auto var_name = last_instr->VarName();
                        cond_ib->pop_back();
                        return var_name;
                    }
                    throw std::runtime_error("unexpected");
                }();

                auto if_true = BuildRecursiveEx(if_expr->IfTrue());
                auto if_false = BuildRecursiveEx(if_expr->IfFalse());

                auto if_block = std::make_shared< IfBlock>(
                    cond_var_name,
                    if_true,
                    if_false);

                auto modulee = std::make_shared<Module>();
                modulee->push_back(cond_ib);
                modulee->push_back(if_block);
                return modulee;
            }
            else
            {
                throw std::domain_error("unexpectded");
            }
        }
        else
        {
            std::unordered_set<std::string> symbols_seen;

            auto IB = std::make_shared<InstructionBlock>();

            auto deps = head->DepthFirstAnySymbolicDependencyAndThis();
            if (deps.DepthFirst.size() > 0)
            {
                for (auto sym : deps.DepthFirst) {
                    if (sym->IsExo())
                        continue;
                    if (symbols_seen.count(sym->Name()) != 0)
                    {
                        continue;
                    }
                    auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(sym)->Expr();
                    IB->Add(std::make_shared<InstructionDeclareVariable>(sym->Name(), expr));
                    symbols_seen.insert(sym->Name());

                }
            }

            std::string aux_name = "result";
            IB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, head));
            IB->Add(std::make_shared< InstructionReturn>(aux_name));


            auto modulee = std::make_shared<Module>();
            modulee->push_back(IB);
            return modulee;

        }
    }

    std::shared_ptr<Operator> ExpandCall(std::shared_ptr<Operator> const& head)
    {
        struct MapCallSite : OperatorTransform {
            virtual std::shared_ptr<Operator> Apply(std::shared_ptr<Operator> const& ptr) {
                auto candidate = ptr->Clone(shared_from_this());
                if (auto as_call = std::dynamic_pointer_cast<Call>(candidate))
                {
                    auto name = "__tmp_call_" + std::to_string((size_t)as_call.get());
                    names_.push_back(name);
                    auto exo = std::make_shared<EndgenousSymbol>(name, candidate);
                    return exo;
                }
                return candidate;
            }
            std::vector<std::string> names_;
        };

        auto result = head->Clone(std::make_shared<MapCallSite>());
        return result;
    }

    struct DebugControlBlockVisitor : ControlBlockVisitor
    {
        size_t indent_ = 0;
        void indent()
        {
            if (indent_ != 0)
            {
                std::cout << std::string(indent_ * 4, ' ');
            }
        }
        void AcceptInstruction(const std::shared_ptr<const Instruction>& instr) override
        {
            indent();
            std::cout << "AcceptInstruction : ";
            instr->EmitCode(std::cout);
            if (auto as_lvalue_assign = std::dynamic_pointer_cast<const InstructionDeclareVariable>(instr))
            {
                if (auto as_call = std::dynamic_pointer_cast<const Call>(as_lvalue_assign->as_operator_()))
                {
                    std::cout << " // is a call site\n";
                }

            }
        }
        void AcceptIf(const std::shared_ptr<const IfBlock>& if_block)
        {
            indent();
            std::cout << "BEGIN IF BLOCK\n";
            ++indent_;
            if_block->IfTrue()->Accept(*this);
            --indent_;
            indent();
            std::cout << "BEGIN ELSE BLOCK\n";
            ++indent_;
            if_block->IfFalse()->Accept(*this);
            --indent_;
            indent();
            std::cout << "END IF BLOCK\n";

        }
        void AcceptCall(const std::shared_ptr<const CallBlock>& call_block)
        {
            indent();
            std::cout << "CALL SITE\n";
        }
    };




    /*
    Creates a linear sequence of instructions, and splits out if statement, and call sites
    */
    struct InstructionLinearizer : ControlBlockVisitor
    {
        void AcceptInstruction(const std::shared_ptr<const Instruction>& instr) override
        {

            if (auto as_lvalue_assign = std::dynamic_pointer_cast<const InstructionDeclareVariable>(instr))
            {
                if (seen_.count(instr))
                {
                    // short circute
                    return;
                }
                auto make_rvalue = [](std::shared_ptr<Operator> const& op)->std::shared_ptr<ProgramCode::RValue>
                {
                    if (auto as_sym = std::dynamic_pointer_cast<const Symbol>(op))
                    {
                        return std::dynamic_pointer_cast<ProgramCode::RValue>(std::make_shared<ProgramCode::LValue>(as_sym->Name()));
                    }
                    if (auto as_lit = std::dynamic_pointer_cast<const Constant>(op))
                    {
                        return std::make_shared<ProgramCode::DoubleConstant>(as_lit->Value());
                    }
                    throw std::domain_error("not an rvalue");
                };

                if (auto as_call = std::dynamic_pointer_cast<const Call>(as_lvalue_assign->as_operator_()))
                {
                    std::vector<std::shared_ptr<ProgramCode::RValue> > arg_list;
                    for (auto const& arg : as_call->Children())
                    {
                        arg_list.push_back(make_rvalue(arg));
                    }
                    auto call_stmt = std::make_shared<ProgramCode::CallStatement>(
                        as_call->FunctionName(),
                        std::vector<std::shared_ptr<ProgramCode::LValue> >{std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName())},
                        arg_list);
                    stmts_.push_back(call_stmt);
                }
                else
                {
                    auto op = as_lvalue_assign->as_operator_();
                    if (auto as_binary = std::dynamic_pointer_cast<const BinaryOperator>(op))
                    {
                        auto left_name = make_rvalue(as_binary->LParam());
                        auto right_name = make_rvalue(as_binary->RParam());

                        auto mapped_op = [&]()->ProgramCode::OpCode
                        {
                            using ProgramCode::OpCode;
                            switch (as_binary->OpKind())
                            {
                            case BinaryOperatorKind::OP_ADD: return OpCode::OP_ADD;
                            case BinaryOperatorKind::OP_SUB: return OpCode::OP_SUB;
                            case BinaryOperatorKind::OP_MUL: return OpCode::OP_MUL;
                            case BinaryOperatorKind::OP_DIV: return OpCode::OP_DIV;
                            case BinaryOperatorKind::OP_POW: return OpCode::OP_POW;
                            case BinaryOperatorKind::OP_MIN: return OpCode::OP_MIN;
                            case BinaryOperatorKind::OP_MAX: return OpCode::OP_MAX;
                            }
                            throw std::domain_error("unknown binary op");
                        }();

                        auto three_address = std::make_shared<ProgramCode::ThreeAddressCode>(
                            mapped_op,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            left_name,
                            right_name);

                        stmts_.push_back(three_address);

                    }
                    else if (auto as_unary = std::dynamic_pointer_cast<const UnaryOperator>(op))
                    {
                        auto mapped_op = [&]()->ProgramCode::OpCode
                        {
                            using ProgramCode::OpCode;
                            switch (as_unary->OpKind())
                            {
                            case UnaryOperatorKind::UOP_USUB: return OpCode::OP_USUB;
                            }
                            throw std::domain_error("unknown unaru op");
                        }();

                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            mapped_op,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            make_rvalue(as_unary->At(0)));

                        stmts_.push_back(two_address);
                    }
                    else if (auto as_sym = std::dynamic_pointer_cast<const Symbol>(op))
                    {
                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            ProgramCode::OpCode::OP_ASSIGN,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            std::make_shared<ProgramCode::LValue>(as_sym->Name()));

                        stmts_.push_back(two_address);
                    }
                    else if (auto as_exp = std::dynamic_pointer_cast<const Exp>(op))
                    {
                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            ProgramCode::OpCode::OP_EXP,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            make_rvalue(as_exp->At(0)));

                        stmts_.push_back(two_address);
                    }
                    else if (auto as_log = std::dynamic_pointer_cast<const Log>(op))
                    {
                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            ProgramCode::OpCode::OP_LOG,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            make_rvalue(as_log->At(0)));

                        stmts_.push_back(two_address);
                    }
                    else if (auto as_phi = std::dynamic_pointer_cast<const Phi>(op))
                    {
                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            ProgramCode::OpCode::OP_PHI,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            make_rvalue(as_phi->At(0)));

                        stmts_.push_back(two_address);
                    }
                    else if (auto as_constant = std::dynamic_pointer_cast<const Constant>(op))
                    {
                        auto two_address = std::make_shared<ProgramCode::TwoAddressCode>(
                            ProgramCode::OpCode::OP_ASSIGN,
                            std::make_shared<ProgramCode::LValue>(as_lvalue_assign->LValueName()),
                            std::make_shared<ProgramCode::DoubleConstant>(as_constant->Value()));

                        stmts_.push_back(two_address);
                    }
                    else
                    {
                        std::string();
                    }
                }
            }
            else if (auto as_return = std::dynamic_pointer_cast<const InstructionReturn>(instr))
            {
                stmts_.push_back(std::make_shared<ProgramCode::ReturnStatement>(
                    std::make_shared<ProgramCode::LValue>(as_return->VarName())));
            }
            else
            {
                throw std::domain_error("unknown type");
            }

            seen_.insert(instr);
        }
        void AcceptIf(const std::shared_ptr<const IfBlock>& if_block)
        {
            InstructionLinearizer if_true;
            InstructionLinearizer if_false;
            if_block->IfTrue()->Accept(if_true);
            if_block->IfFalse()->Accept(if_false);

            auto result = std::make_shared<ProgramCode::IfStatement>(
                std::make_shared<ProgramCode::LValue>(if_block->ConditionVariable()),
                std::make_shared<ProgramCode::StatementList>(if_true.stmts_),
                std::make_shared<ProgramCode::StatementList>(if_false.stmts_));
            stmts_.push_back(result);
        }
        void AcceptCall(const std::shared_ptr<const CallBlock>& call_block)
        {

        }
        std::unordered_set<std::shared_ptr<const Instruction> > seen_;
        std::vector<std::shared_ptr<ProgramCode::Statement> > stmts_;
    };


    struct ExecutionContext
    {
        std::vector<std::string> exo_names;
        std::vector<std::unordered_map<std::string, size_t> > alloc_map_list;
        std::vector<std::shared_ptr<ImpliedMatrixFunction> > jacobian_list;
    };


    std::shared_ptr<ProgramCode::Statement> CloneStmtWithDiffs(
        ExecutionContext& context,
        std::shared_ptr<ProgramCode::Statement> const& stmt)
    {

        using namespace ProgramCode;

        if (auto stmts = std::dynamic_pointer_cast<StatementList>(stmt))
        {
            std::vector<std::shared_ptr<Statement> > new_stmts;
            for (auto const& child_stmt : *stmts)
            {
                auto result = CloneStmtWithDiffs(
                    context,
                    child_stmt);
                new_stmts.push_back(result);

            }
            return std::make_shared<StatementList>(std::vector<std::shared_ptr<Statement> >{new_stmts});
        }
        else if (auto three_addr = std::dynamic_pointer_cast<ThreeAddressCode>(stmt))
        {
            std::string lvalue = three_addr->name_->ToString();
            auto alloc_map = context.alloc_map_list.back();
            if (alloc_map.count(lvalue) == 0)
            {
                auto expr = three_addr->to_operator();


                auto slot = alloc_map.size();
                alloc_map[lvalue] = slot;

                auto matrix_func = ImpliedMatrixFunction::MakeFromOperator(context.jacobian_list.size(), expr);
                context.alloc_map_list.push_back(alloc_map);
                context.jacobian_list.push_back(matrix_func);
            }
            return stmt;
        }
        else if (auto two_addr = std::dynamic_pointer_cast<TwoAddressCode>(stmt))
        {
            auto lvalue = two_addr->rvalue_->ToString();
            auto alloc_map = context.alloc_map_list.back();
            if (alloc_map.count(lvalue) == 0)
            {
                auto expr = two_addr->to_operator();
                auto slot = alloc_map.size();
                alloc_map[lvalue] = slot;
                if (alloc_map.size() != slot + 1)
                {
                    std::string();
                }


                auto matrix_func = ImpliedMatrixFunction::MakeFromOperator(context.jacobian_list.size(), expr);
                context.alloc_map_list.push_back(alloc_map);
                context.jacobian_list.push_back(matrix_func);
            }
            return stmt;
        }
        else if (auto if_stmt = std::dynamic_pointer_cast<IfStatement>(stmt))
        {
            ExecutionContext true_context(context);
            ExecutionContext false_context(context);
            auto true_stmt = CloneStmtWithDiffs(true_context, if_stmt->if_true_);
            auto false_stmt = CloneStmtWithDiffs(true_context, if_stmt->if_false_);
            return std::make_shared<IfStatement>(
                if_stmt->condition_, true_stmt, false_stmt);

        }
        else if (auto call_stmt = std::dynamic_pointer_cast<CallStatement>(stmt))
        {
            auto mapped_result_list = call_stmt->result_list_;
            if (mapped_result_list.size() != 1)
            {
                throw std::domain_error("unexpected");
            }
            for (auto const& arg : call_stmt->arg_list_)
            {
                std::string d_name = "d_" + std::to_string((size_t)call_stmt.get()) + "_" + arg->ToString();
                mapped_result_list.push_back(std::make_shared<LValue>(d_name));
            }



            const auto id = context.jacobian_list.size();
            std::vector<std::shared_ptr<Symbol> > args;
            std::vector<std::shared_ptr<Operator> > diffs;
            for (auto const& arg : call_stmt->arg_list_)
            {
                args.push_back(std::make_shared<ExogenousSymbol>(arg->ToString()));
            }
            for (size_t idx = 1; idx < mapped_result_list.size(); ++idx)
            {
                auto const& d = mapped_result_list[idx];
                diffs.push_back(std::make_shared<ExogenousSymbol>(d->ToString()));
            }
            auto jacoibian = std::make_shared< ImpliedMatrixFunction>(id, "dummy", args, diffs);

            auto lvalue = call_stmt->result_list_[0]->ToString();
            auto alloc_map = context.alloc_map_list.back();
            auto slot = alloc_map.size();
            alloc_map[lvalue] = slot;


            context.alloc_map_list.push_back(alloc_map);
            context.jacobian_list.push_back(jacoibian);

            return std::make_shared<CallStatement>(
                call_stmt->function_name_,
                mapped_result_list,
                call_stmt->arg_list_);
        }
        else if (auto return_stmt = std::dynamic_pointer_cast<ReturnStatement>(stmt))
        {
            // HERE we add the the jacobian


#if 0
            for (auto const& func : context.jacobian_list)
            {
                func->PrintDebug();
            }
#endif

            std::vector<std::shared_ptr<SymbolicMatrix> > adj_matrix_list;
            for (size_t idx = context.jacobian_list.size(); idx != 0; )
            {
                bool is_terminal = (idx == context.jacobian_list.size());
                --idx;

                auto const& alloc_map = context.alloc_map_list[idx];

                auto matrix_decl = context.jacobian_list[idx]->MakeMatrix(alloc_map, is_terminal);

                adj_matrix_list.push_back(matrix_decl->Matrix());

            }

            auto fold_backwards = [&]()
            {
                // (M1*(M2*M3))
                std::shared_ptr<SymbolicMatrix> adj_matrix = adj_matrix_list[0];
                for (size_t idx = 1; idx < adj_matrix_list.size(); ++idx)
                {
                    adj_matrix = adj_matrix_list[idx]->Multiply(*adj_matrix);
                }
                return adj_matrix;
            };

            auto adj_matrix = fold_backwards();

            auto three_address_transform = std::make_shared<Transform::RemapUnique>("__adj");

            std::unordered_set<std::string> three_addr_seen;
#if 0
            for (auto const& p : context.alloc_map_list.back())
            {
                three_addr_seen.insert(p.first);
            }
#endif

            auto AADIB = std::make_shared<InstructionBlock>();

            std::vector<std::shared_ptr<RValue> > output_list;
            output_list.push_back(return_stmt->value_);


            for (auto const& exo : context.exo_names)
            {
                auto d_sym = std::string("d_") + exo;

                auto slot = context.alloc_map_list[0].find(exo)->second;
                auto d_expr_orig = adj_matrix->At(slot, 0);

                Transform::FoldZero fold_zero;
                auto d_expr_no_zero = fold_zero.Fold(d_expr_orig);
                auto d_expr_three_address = d_expr_no_zero->Clone(three_address_transform);

                auto deps = d_expr_three_address->DepthFirstAnySymbolicDependencyAndThis();
                for (auto head : deps.DepthFirst) {
                    if (head->IsExo())
                        continue;
                    if (three_addr_seen.count(head->Name()) == 0) {
                        auto expr = std::reinterpret_pointer_cast<EndgenousSymbol>(head)->Expr();
                        AADIB->Add(std::make_shared<InstructionDeclareVariable>(head->Name(), expr));
                        three_addr_seen.insert(head->Name());
                    }

                }

                std::string aux_name = "result_" + d_sym;
                output_list.push_back(std::make_shared<LValue>(aux_name));
                AADIB->Add(std::make_shared<InstructionDeclareVariable>(aux_name, d_expr_three_address));


            }

            auto l = std::make_shared<InstructionLinearizer>();
            AADIB->Accept(*l);

            auto stmts_with_aad = l->stmts_;
            stmts_with_aad.push_back(std::make_shared<ReturnArrayStatement>(output_list));
            return std::make_shared<StatementList>(stmts_with_aad);
        }
        else
        {
            throw std::domain_error("todo");
        }

    }


    std::shared_ptr<ProgramCode::Function> CloneWithDiffs(std::shared_ptr<ProgramCode::Function>& f)
    {
        auto stmts = f->Statements();
        std::unordered_map<std::string, size_t> alloc_map;
        for (auto const& arg : f->Args())
        {
            auto slot = alloc_map.size();
            alloc_map[arg] = slot;
        }
        ExecutionContext context;
        context.exo_names = f->Args();
        context.alloc_map_list.push_back(alloc_map);

        auto mapped_stmts = CloneStmtWithDiffs(context, stmts);
        return std::make_shared<ProgramCode::Function>(
            f->Name(),
            f->Args(),
            std::vector< std::shared_ptr<ProgramCode::Statement>>{ mapped_stmts });
    }



    template<class Kernel>
    void PrintCode()
    {
        auto ad_kernel = typename Kernel::template Build<DoubleKernel>();

        auto arguments = ad_kernel.Arguments();

        std::vector<DoubleKernel> symbolc_arguments;
        for (auto const& arg : arguments)
        {
            symbolc_arguments.push_back(DoubleKernel::BuildFromExo(arg));
        }
        auto function_root = ad_kernel.EvaluateVec(symbolc_arguments);

        auto head = function_root.as_operator_();

        auto three_address_transform = std::make_shared<Transform::RemapUnique>();
        auto three_address_tree = head->Clone(three_address_transform);

        //three_address_transform->Debug();

        //auto call_expanded_head = ExpandCall(three_address_tree);
        auto call_expanded_head = three_address_tree;

        auto block = BuildRecursiveEx(call_expanded_head);

        auto f = std::make_shared<Function>(block);
        for (auto const& arg : arguments)
        {
            f->AddArg(std::make_shared<FunctionArgument>(FAK_Double, arg));
        }

        //f->GetModule()->EmitCode(std::cout);

        auto M = f->GetModule();

        auto v = std::make_shared< DebugControlBlockVisitor>();
        //M->Accept(*v);

        auto l = std::make_shared< InstructionLinearizer>();
        M->Accept(*l);

        auto ff = std::make_shared<ProgramCode::Function>(ad_kernel.Name(), arguments, l->stmts_);
        //ff->DebugPrint();

        //ProgramCode::CodeWriter{}.EmitCode(std::cout, ff);

        auto g = CloneWithDiffs(ff);

        ProgramCode::CodeWriter{}.EmitCode(std::cout, g);
    }


} // end namespace CodeGen
} // end namespace Cady


#endif // INCLUDE_CADY_CODEGEN_H
