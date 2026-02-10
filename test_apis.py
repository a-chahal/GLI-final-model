#!/usr/bin/env python3
"""Test and diagnose broken APIs for compound collection."""
import requests
import time
import json

print("=" * 70)
print("TEST 1: PubChem - Small batch GET (5 CIDs)")
print("=" * 70)
test_cids = [2244, 3672, 5090, 2519, 5281]
cid_str = ",".join(str(c) for c in test_cids)
url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,MolecularWeight,XLogP/JSON"
print(f"URL length: {len(url1)}")
r1 = requests.get(url1, timeout=30)
print(f"Status: {r1.status_code}")
if r1.status_code == 200:
    data = r1.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    print(f"Returned {len(props)} compounds")

print()
print("=" * 70)
print("TEST 2: PubChem - AID 2551 active CIDs")
print("=" * 70)
time.sleep(0.5)
r2 = requests.get(
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/2551/cids/JSON",
    params={"cids_type": "active"}, timeout=30
)
print(f"Status: {r2.status_code}")
all_cids = []
if r2.status_code == 200:
    data2 = r2.json()
    info_list = data2.get("InformationList", {}).get("Information", [])
    for info in info_list:
        all_cids.extend(info.get("CID", []))
    print(f"Active CIDs: {len(all_cids)}")

    # Now test batch of 100 (smaller)
    time.sleep(0.5)
    batch = all_cids[:100]
    batch_str = ",".join(str(c) for c in batch)
    url3 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{batch_str}/property/CanonicalSMILES/JSON"
    print(f"\nBatch of 100 URL length: {len(url3)}")
    r3 = requests.get(url3, timeout=60, headers={"Accept": "application/json"})
    print(f"Batch GET status: {r3.status_code}")
    if r3.status_code == 200:
        data3 = r3.json()
        props3 = data3.get("PropertyTable", {}).get("Properties", [])
        print(f"Returned {len(props3)} compounds")
    else:
        print(f"Error response: {r3.text[:300]}")

    # Test POST method for large batches
    time.sleep(0.5)
    batch200 = all_cids[:200]
    batch200_str = ",".join(str(c) for c in batch200)
    url4 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES/JSON"
    r4 = requests.post(url4, data={"cid": batch200_str}, timeout=60)
    print(f"\nPOST batch of 200 status: {r4.status_code}")
    if r4.status_code == 200:
        data4 = r4.json()
        props4 = data4.get("PropertyTable", {}).get("Properties", [])
        print(f"POST returned {len(props4)} compounds")
    else:
        print(f"Error: {r4.text[:300]}")
else:
    # Try the "all" CIDs approach
    time.sleep(0.5)
    r2b = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/2551/cids/JSON",
        timeout=30
    )
    print(f"All CIDs status: {r2b.status_code}")
    if r2b.status_code == 200:
        data2b = r2b.json()
        info_list = data2b.get("InformationList", {}).get("Information", [])
        for info in info_list:
            all_cids.extend(info.get("CID", []))
        print(f"All CIDs: {len(all_cids)}")

print()
print("=" * 70)
print("TEST 3: DGIdb v5 GraphQL API")
print("=" * 70)
# DGIdb 5.0 uses GraphQL
dgidb_url = "https://dgidb.org/api/graphql"
query = """
{
  genes(names: ["GLI1", "GLI2", "SMO", "PTCH1", "SHH"]) {
    nodes {
      name
      interactions {
        nodes {
          drug {
            name
            conceptId
          }
          interactionScore
          interactionTypes {
            type
          }
        }
      }
    }
  }
}
"""
r5 = requests.post(dgidb_url, json={"query": query}, timeout=30)
print(f"Status: {r5.status_code}")
if r5.status_code == 200:
    data5 = r5.json()
    total_interactions = 0
    for gene_node in data5.get("data", {}).get("genes", {}).get("nodes", []):
        gene_name = gene_node.get("name", "?")
        interactions = gene_node.get("interactions", {}).get("nodes", [])
        total_interactions += len(interactions)
        drugs = [i.get("drug", {}).get("name", "?") for i in interactions[:5]]
        print(f"  {gene_name}: {len(interactions)} drug interactions (sample: {drugs[:3]})")
    print(f"  Total interactions: {total_interactions}")
else:
    print(f"Error: {r5.text[:300]}")

print()
print("=" * 70)
print("TEST 4: ZINC22 (new API)")
print("=" * 70)
# Test ZINC22 Cartblanche API
zinc22_url = "https://cartblanche22.docking.org/search/similarity"
r6 = requests.get(zinc22_url, timeout=10)
print(f"ZINC22 API reachable: {r6.status_code}")

# Test ZINC15 (legacy, might still work)
zinc15_url = "https://zinc15.docking.org/substances/search.json"
try:
    r7 = requests.get(zinc15_url, params={"smiles": "c1ccccc1", "type": "sim", "threshold": 0.7, "count": 5}, timeout=10)
    print(f"ZINC15 API: {r7.status_code}")
except Exception as e:
    print(f"ZINC15 API error: {e}")

print()
print("=" * 70)
print("TEST 5: ChEMBL - Broader zinc finger query")
print("=" * 70)
# Search for ALL zinc finger protein targets, not just Hh pathway
chembl_url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
r8 = requests.get(chembl_url, params={"q": "zinc finger", "limit": 5}, timeout=30,
                   headers={"Accept": "application/json"})
print(f"ChEMBL zinc finger targets: {r8.status_code}")
if r8.status_code == 200:
    data8 = r8.json()
    targets = data8.get("targets", [])
    print(f"  Found {len(targets)} targets (of {data8.get('page_meta', {}).get('total_count', '?')} total)")
    for t in targets[:5]:
        print(f"    {t.get('target_chembl_id')}: {t.get('pref_name', '?')[:60]}")

# Also check how many compounds exist for zinc finger targets
time.sleep(0.3)
r9 = requests.get(chembl_url, params={"q": "zinc finger", "limit": 100}, timeout=30,
                   headers={"Accept": "application/json"})
if r9.status_code == 200:
    data9 = r9.json()
    total_zf = data9.get("page_meta", {}).get("total_count", 0)
    print(f"  Total zinc finger targets in ChEMBL: {total_zf}")

print()
print("=" * 70)
print("TEST 6: DrugBank open data")
print("=" * 70)
# DrugBank has an open-access subset
drugbank_url = "https://go.drugbank.com/releases/latest#open-data"
print(f"DrugBank open data: manual download from {drugbank_url}")
print("  Contains ~2800 FDA-approved drugs with SMILES")
print("  Good for drug repurposing screen")
