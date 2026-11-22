from analytic_evaluation import install as install_parent


def install(original):
    install_parent(original)
    evaluate=original.evaluate

    def tracked(model,dataset,focal,args,device,epoch):
        result=evaluate(model,dataset,focal,args,device,epoch)
        state=model.correction.analytic
        result['paired_supervision']=dict(eligible_class_counts=state.eligible_counts.cpu().tolist(),
            paired_positives=state.paired_positives.item(),pairs=state.pair_count.item(),
            solved_pairs=state.last_solved.item(),eligible_domains=state.support.sum().item(),
            history_rows=(state.negative_group>=0).sum().item())
        return result
    original.evaluate=tracked
