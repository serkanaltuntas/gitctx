from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
from gitctx.reviewed_job import BudgetStop, Guard, GuardedDataset, prediction_metrics, review_accuracy

class ReviewedJobTests(TestCase):
    def test_budget_and_disk_guards_run_before_dataset_access(self):
        class Dataset:
            fingerprint='f'
            def ids(self,p):return ('a',)
            def example(self,*args,**kwargs):raise AssertionError('must not access data after deadline')
        with patch('gitctx.reviewed_job.time.time',return_value=100):
            ds=GuardedDataset(Dataset(),Guard('.',100,8))
            with self.assertRaisesRegex(BudgetStop,'wall_time'):ds.example('a',partition='train')
        with patch('gitctx.reviewed_job.time.time',return_value=99), patch('gitctx.reviewed_job.shutil.disk_usage',return_value=SimpleNamespace(free=7)):
            with self.assertRaisesRegex(BudgetStop,'disk_free'):Guard('.',100,8)()

    def test_invalid_and_empty_predictions_count_toward_collapse_denominator(self):
        p=[dict(record_id='a',message='fix: correct index',decode_error=None,stop_reason='stop_token'),
           dict(record_id='b',message=None,decode_error='UnicodeDecodeError',stop_reason='token_limit'),
           dict(record_id='c',message='',decode_error=None,stop_reason='stop_token')]
        m=prediction_metrics(p,{'a':'fix: correct index','b':'docs: clarify input','c':'test: cover input'})
        self.assertEqual(m['records'],3);self.assertEqual(m['format_valid'],1)
        self.assertEqual(m['decode_errors'],1);self.assertEqual(m['dominant_header_count'],2)

    def test_factuality_review_requires_exact_panel_prediction_and_source_binding(self):
        panel={'entries':[{'record_id':'a'}]};records={'a':{'diff':'+new\n-old\n'}}
        review={'predictions_sha256':'p','panel_sha256':'q','reviewer_kind':'assistant','independent_human_review':False,
                'entries':[{'record_id':'a','correct_principal_change':True,'unsupported_claim':False,
                            'note':'Source supports output','evidence':[{'line':1,'quote':'+new\n'}]}]}
        opts=dict(predictions_sha256='p',panel_sha256='q',panel=panel,records=records)
        self.assertEqual(review_accuracy(review,**opts),1.)
        with self.assertRaisesRegex(ValueError,'lineage'):review_accuracy({**review,'predictions_sha256':'changed'},**opts)
        with self.assertRaisesRegex(ValueError,'every frozen'):review_accuracy({**review,'entries':[]},**opts)
        review['entries'][0]['evidence'][0]['quote']='invented'
        with self.assertRaisesRegex(ValueError,'differs'):review_accuracy(review,**opts)
