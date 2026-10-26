import copy
import unittest

from factorial_results import compare,gains
from experiment_provenance import CUDA_SOURCES


class FactorialTests(unittest.TestCase):
    def test_combined_percentages_are_not_added(self):
        values=gains(100,90,80,72)
        self.assertAlmostEqual(values["our_gain_stock_pct"],10)
        self.assertAlmostEqual(values["our_gain_unlocked_pct"],10)
        self.assertAlmostEqual(values["unlock_gain_vendor_pct"],20)
        self.assertAlmostEqual(values["combined_gain_pct"],28)

    def fixture(self):
        required=set(CUDA_SOURCES)|{"native/driver.cpp","native/build/libkernel_lab.so","bench.py","cublas_baseline.py"}
        timing=lambda us:{"median_us":us,"samples_us":[us]*9}
        return {"kind":"controlled-comparison","status":"complete","schema_version":2,"run_id":"stock",
                "source_sha256":{name:"same" for name in required},
                "environment":{"torch":"test","cuda":"test","cublas_version":1,"compilation_cache_key":"same",
                "inventory":{"gpu":"same UUID/driver","kernel":"same","pci_address":"0000:01:00.0",
                "module_loaded":False,"unlock_boot_option_present":False,"unlock_success_logged":False,
                "post_flr_selectors_logged":False}},
                "parameters":{"rounds":15},"cases":[{"shape_mnk":[1,64,64],"seed":1,"best_vendor_us":100,
                "best_vendor_method":"torch_mm","bitwise_round_trip":True,"aa_time_saved_pct":0,
                "methods":{"torch_mm":{"cold":timing(100)},
                "lossless_fp32":{"cold":timing(90),"meets_precision_requirement":True,"output_relative_l2_error":1e-7}}}]}

    def pair(self):
        stock=self.fixture()
        after=copy.deepcopy(stock)
        after["run_id"]="unlocked"
        for key in ("module_loaded","unlock_boot_option_present","unlock_success_logged","post_flr_selectors_logged"):
            after["environment"]["inventory"][key]=True
        after["cases"][0]["best_vendor_us"]=80
        after["cases"][0]["methods"]["torch_mm"]["cold"]={"median_us":80,"samples_us":[80]*9}
        after["cases"][0]["methods"]["lossless_fp32"]["cold"]={"median_us":72,"samples_us":[72]*9}
        return stock,after

    def test_matched_reports_and_confounds(self):
        stock,after=self.pair()
        self.assertAlmostEqual(compare(stock,after)[0]["combined_gain_pct"],28)
        for key,value in (("source_sha256",{"kernel":"changed"}),("parameters",{"rounds":3}),
                          ("kind","tensorcore-diagnostic"),("cases",[])):
            broken=copy.deepcopy(after)
            broken[key]=value
            with self.assertRaises(ValueError): compare(stock,broken)

    def test_duplicate_cases_and_identical_run_rejected(self):
        stock,after=self.pair()
        after["cases"]*=2
        with self.assertRaises(ValueError): compare(stock,after)
        stock,after=self.pair()
        after["run_id"]=stock["run_id"]
        with self.assertRaises(ValueError): compare(stock,after)

    def test_vendor_unlock_effect_holds_the_vendor_route_fixed(self):
        stock,after=self.pair()
        case=after["cases"][0]
        case["methods"]["cublas_mm"]={"cold":{"median_us":70,"samples_us":[70]*9}}
        case["best_vendor_method"]="cublas_mm"
        case["best_vendor_us"]=70
        row=compare(stock,after)[0]
        self.assertAlmostEqual(row["unlock_gain_vendor_pct"],20)
        self.assertLess(row["our_gain_unlocked_pct"],0)

    def test_no_unlock_evidence_no_attribution(self):
        stock,after=self.pair()
        after["environment"]["inventory"]["post_flr_selectors_logged"]=False
        with self.assertRaises(ValueError): compare(stock,after)
