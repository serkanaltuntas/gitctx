from pathlib import Path
from copy import deepcopy
import hashlib
import json
import tempfile
from unittest import TestCase

from gitctx import reviewed_inputs as ri, window_mixture as wm
from gitctx.reference_overlay import artifact_hash
import test_reviewed_dataset as fixtures


class ReviewedInputTests(TestCase):
    def setUp(self):
        self.fixture = fixtures.ReviewedDatasetTests(); self.fixture.setUp()
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name); self.manifest = {'version':ri.VERSION,'files':{}}
        f = self.fixture
        self.put('source', [f.train,f.validation], lines=True)
        self.put('coverage', f.coverage, lines=True)
        f.tokenizer.save(self.root/'tokenizer.json'); self.pin('tokenizer','tokenizer.json')
        self.put('split_protocol', {'validation_ids':['validation']})
        self.put('student_manifest', {'inputs':{
            self.manifest['files']['source']['path']:self.manifest['files']['source']['sha256'],
            self.manifest['files']['split_protocol']['path']:self.manifest['files']['split_protocol']['sha256']},
            'outputs':{'tokenizer.json':self.manifest['files']['tokenizer']['sha256'],
                       'coverage.jsonl':self.manifest['files']['coverage']['sha256']},
            'tokenizer_fit_ids':['train']})
        self.put('review_selection', [{'record_id':'validation'}], lines=True)
        self.put('artifact', [f.artifact], lines=True)
        self.put('review_index', [{**f.index,'artifact_file':self.manifest['files']['artifact']['path']}], lines=True)
        windows,groups=[],[]
        for record in (f.train,f.validation):
            ws=wm.ew.build_windows(record,f.tokenizer)
            for w in ws:w['partition']=record['id']
            windows+=ws
            g=wm.prepare(record,f.tokenizer,windows=ws)
            groups.append(dict(record_id=record['id'],partition=record['id'],target_assigned=False,
                               group_sha256=artifact_hash(g),window_ids=g['window_ids']))
        self.put('windows',windows,lines=True);self.put('groups',groups,lines=True)

    def pin(self, name, filename):
        self.manifest['files'][name]={'path':filename,'sha256':hashlib.sha256((self.root/filename).read_bytes()).hexdigest()}

    def put(self, name, value, lines=False):
        filename=name+('.jsonl' if lines else '.json')
        (self.root/filename).write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value))
        self.pin(name,filename)

    def test_complete_pinned_input_join_preserves_target_and_partition(self):
        ds=ri.load_dataset(self.root,self.manifest)
        self.assertEqual(ds.ids('train'),('train',));self.assertEqual(ds.ids('validation'),('validation',))
        e=ds.example('validation',partition='validation')
        self.assertEqual(e['reference']['target_message'],self.fixture.artifact['replacement_message'])

    def test_unpinned_artifact_source_mutation_and_frozen_selection_mismatch_fail(self):
        m=deepcopy(self.manifest);del m['files']['artifact']
        with self.assertRaisesRegex(ValueError,'not pinned'):ri.load_dataset(self.root,m)
        m=deepcopy(self.manifest);m['files']['source']['sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):ri.load_dataset(self.root,m)
        self.put('review_selection',[],lines=True)
        with self.assertRaisesRegex(ValueError,'frozen selection'):ri.load_dataset(self.root,self.manifest)

    def test_split_cannot_be_changed_by_rehashing_coverage_alone(self):
        altered=[dict(record_id='train',partition='validation'),dict(record_id='validation',partition='train')]
        self.put('coverage',altered,lines=True)
        student=json.loads((self.root/'student_manifest.json').read_text())
        student['outputs']['coverage.jsonl']=self.manifest['files']['coverage']['sha256']
        self.put('student_manifest',student)
        with self.assertRaisesRegex(ValueError,'frozen active partition'):ri.load_dataset(self.root,self.manifest)
