import copy
import unittest
from gitctx.proof_input_diagnostic import (variant_record, replace_diff, select_diverse,
    choose_donors, sequence_for_prompt, retained_changes, summarize_predictions)
from gitctx.proof_tokenizer import tokenize_text


def record(i=0, repo='a', diff='-old\n+new\n'):
    return {'id':str(i),'source_repo_url':repo,'diff':diff,'target_message':'fix: change',
        'messages':[{'role':'system','content':'Write a commit.'},
            {'role':'user','content':'Repo: '+repo+'\nDiff:\n'+diff+'\nReturn JSON'},
            {'role':'assistant','content':'fix: change'}]}

class InputDiagnosticTests(unittest.TestCase):
    def test_diff_replacement_preserves_metadata_suffix_and_source(self):
        r=record();original=copy.deepcopy(r);d=record(1,'b','+donor\n')
        changed=variant_record(r,d,'swapped_diff')
        self.assertEqual(changed['messages'][1]['content'],'Repo: a\nDiff:\n+donor\n\nReturn JSON')
        self.assertEqual(r,original)
        self.assertEqual(changed['messages'][2],r['messages'][2])
        whole=variant_record(r,d,'swapped_user')
        self.assertEqual(whole['messages'][1],d['messages'][1])
        self.assertEqual(whole['messages'][2],r['messages'][2])
    def test_misaligned_diff_fails_instead_of_silent_replacement(self):
        r=record();r['diff']='other'
        with self.assertRaises(ValueError):replace_diff(r,'')
    def test_sampling_is_order_independent_and_donors_cross_repositories(self):
        rows=[record(i,'a' if i%2 else 'b') for i in range(16)]
        first=select_diverse(rows,8)
        self.assertEqual(first,select_diverse(rows[::-1],8))
        donors=choose_donors(first);by_id={r['id']:r for r in first}
        self.assertEqual(len(set(r['id'] for r in first)),8)
        for i,d in donors.items():self.assertNotEqual(by_id[i]['source_repo_url'],by_id[d]['source_repo_url'])
    def test_loss_is_only_on_target_and_terminals(self):
        vocab={t:i for i,t in enumerate(['<unk>','<bos>','<assistant>','fix',':','change','<sep>','<eos>'])}
        seq=sequence_for_prompt(['<bos>','<assistant>'],'fix: change',vocab)
        self.assertEqual(seq['loss_mask'],[0,0,1,1,1,1,1])
        self.assertEqual(seq['input_ids'],[1,2,3,4,5,6,7])
        with self.assertRaises(ValueError):sequence_for_prompt(['<bos>']*8192,'fix: change',vocab)
    def test_changed_line_retention_counts_middle_removal(self):
        r=record(diff='--- a/x\n+++ b/x\n-old\n+new\n')
        full=tokenize_text(r['messages'][1]['content'])
        self.assertEqual(retained_changes(r,full)['fully_retained'],2)
        self.assertEqual(retained_changes(r,full[:2])['fully_removed'],2)
    def test_paired_metrics_do_not_equate_changed_output_with_accuracy(self):
        rows=[]
        for variant,ce,out in [('real',2,[1]),('empty_diff',3,[1]),('swapped_diff',4,[2]),('swapped_user',1,[3])]:
            rows.append({'record_id':'a','variant':variant,'ce':ce,'loss_tokens':3,'output_token_ids':out,
                'format_valid':True,'type_match':False,'scope_match':False,'exact_text':False,'pair':['fix',None]})
        result=summarize_predictions(rows)
        self.assertEqual(result['empty_diff']['identical_to_real'],1)
        self.assertEqual(result['swapped_diff']['mean_paired_ce_increase'],2)
        self.assertEqual(result['swapped_user']['real_lower_ce'],0)
        self.assertEqual(result['swapped_diff']['type_match'],0)

if __name__=='__main__':unittest.main()
