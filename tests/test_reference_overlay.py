import json
import unittest
from copy import deepcopy
from gitctx.reference_overlay import artifact_hash,build_override
from gitctx.reference_review import sha
from test_student_input import record

class ReferenceOverlayTests(unittest.TestCase):
    def setUp(self):
        self.r=record();self.r['data_split']='DEV'
        fields={'type':'fix','scope':'a','subject':'use new value'};target='fix(a): use new value'
        self.c={'record_id':self.r['id'],'source_diff_sha256':sha(self.r['diff']),
            'target_origin':'licensed_open_teacher','model_license':'Apache-2.0','validation_errors':[],
            'model':'test','model_digest':'abc','target':target,'target_sha256':sha(target),
            'fields':fields,'response':{'response':json.dumps(fields)}}
        self.a={'decision':'accept','method':'full_diff_review','reviewer_kind':'assistant',
            'candidate_sha256':artifact_hash(self.c),'source_diff_sha256':sha(self.r['diff']),
            'timestamp':'2026-01-01','note':'The value changed.',
            'claims':[{'text':target,'decision':'supported','evidence':[{'line':6,'quote':'+new'}]}]}
        class T:
            def encode(self,text):return [1]*10
        self.t=T()
    def build(self,r=None,c=None,a=None):
        return build_override(r or self.r,c or self.c,a or self.a,self.t,teacher_revision='revision')
    def test_explicit_review_creates_overlay_without_rewriting_original(self):
        original=deepcopy(self.r);out=self.build()
        self.assertEqual(self.r,original)
        self.assertEqual(out['replacement_message'],self.c['target'])
        self.assertFalse(out['independent_human_review'])
        self.assertEqual(out['verification_kind'],'assistant')
    def test_reviewer_cannot_substitute_new_text_for_teacher_output(self):
        c=deepcopy(self.c);c['target']='fix(a): another value';c['target_sha256']=sha(c['target'])
        a=deepcopy(self.a);a['candidate_sha256']=artifact_hash(c)
        with self.assertRaisesRegex(ValueError,'differs from teacher'):self.build(c=c,a=a)
    def test_no_report_controls_or_unreviewed_candidates(self):
        for r in ({**self.r,'data_split':'REPORT'},{**self.r,'evaluation_only':True}):
            with self.assertRaises(ValueError):self.build(r=r)
        for a in ({**self.a,'decision':'model_accept'},{**self.a,'candidate_sha256':'wrong'},
                  {**self.a,'claims':[]},{**self.a,'reviewer_kind':'automatic_parser'}):
            with self.assertRaises(ValueError):self.build(a=a)
    def test_evidence_must_be_actual_changed_source(self):
        for evidence in ([{'line':1,'quote':'diff --git'}],[{'line':6,'quote':'imagined'}]):
            a=deepcopy(self.a);a['claims'][0]['evidence']=evidence
            with self.assertRaises(ValueError):self.build(a=a)
