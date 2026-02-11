"""
Batch Docking Script for GLI1 Zinc Finger Regions
Docks multiple compounds to both ZF2-3 (E119/E167) and ZF4-5 (K340/K350) regions
"""

import os
import sys
import csv
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from vina import Vina
import numpy as np
from Bio.PDB import PDBParser
from datetime import datetime

class DualSiteBatchDocker:
    def __init__(self, receptor_pdb='gli_structure/2gli_with_zinc.pdb', 
                 receptor_pdbqt='gli_structure/2gli_receptor.pdbqt',
                 output_dir='docking_results'):
        
        self.receptor_pdb = receptor_pdb
        self.receptor_pdbqt = receptor_pdbqt
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each site
        self.zf23_dir = self.output_dir / 'ZF2-3'
        self.zf45_dir = self.output_dir / 'ZF4-5'
        self.zf23_dir.mkdir(exist_ok=True)
        self.zf45_dir.mkdir(exist_ok=True)
        
        # Initialize binding sites
        self._setup_binding_sites()
        
        print("="*70)
        print("GLI1 DUAL-SITE BATCH DOCKING")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"ZF2-3 results: {self.zf23_dir}")
        print(f"ZF4-5 results: {self.zf45_dir}")
        
    def _setup_binding_sites(self):
        """Setup binding sites for both zinc finger regions"""
        print("\n[SETUP] Analyzing receptor structure...")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('GLI', self.receptor_pdb)
        
        # ZF2-3 Site: E119 and E167
        print("\n  ZF2-3 Site (E119/E167):")
        e119_coord = None
        e167_coord = None
        
        # ZF4-5 Site: K340 and K350
        print("  ZF4-5 Site (K340/K350):")
        k340_coord = None
        k350_coord = None
        
        for chain in structure[0]:
            if chain.id == 'A':
                for res in chain:
                    if res.id[0] == ' ':
                        try:
                            ca_coord = res['CA'].coord
                            
                            # Check for ZF2-3 residues
                            if res.resname == 'GLU' and res.id[1] == 119:
                                e119_coord = ca_coord
                                print(f"    Found E119 at ({ca_coord[0]:.1f}, {ca_coord[1]:.1f}, {ca_coord[2]:.1f})")
                            elif res.resname == 'GLU' and res.id[1] == 167:
                                e167_coord = ca_coord
                                print(f"    Found E167 at ({ca_coord[0]:.1f}, {ca_coord[1]:.1f}, {ca_coord[2]:.1f})")
                            
                            # Check for ZF4-5 residues
                            elif res.resname == 'LYS' and res.id[1] == 340:
                                k340_coord = ca_coord
                                print(f"    Found K340 at ({ca_coord[0]:.1f}, {ca_coord[1]:.1f}, {ca_coord[2]:.1f})")
                            elif res.resname == 'LYS' and res.id[1] == 350:
                                k350_coord = ca_coord
                                print(f"    Found K350 at ({ca_coord[0]:.1f}, {ca_coord[1]:.1f}, {ca_coord[2]:.1f})")
                        except:
                            pass
        
        # Calculate ZF2-3 center
        if e119_coord is not None and e167_coord is not None:
            center_23 = (e119_coord + e167_coord) / 2
            self.zf23_center = [float(center_23[0]), float(center_23[1]), float(center_23[2])]
            print(f"  ✓ ZF2-3 center: ({self.zf23_center[0]:.1f}, {self.zf23_center[1]:.1f}, {self.zf23_center[2]:.1f})")
        else:
            raise ValueError("Could not find E119/E167 for ZF2-3 site")
        
        # Calculate ZF4-5 center
        if k340_coord is not None and k350_coord is not None:
            center_45 = (k340_coord + k350_coord) / 2
            self.zf45_center = [float(center_45[0]), float(center_45[1]), float(center_45[2])]
            print(f"  ✓ ZF4-5 center: ({self.zf45_center[0]:.1f}, {self.zf45_center[1]:.1f}, {self.zf45_center[2]:.1f})")
        else:
            raise ValueError("Could not find K340/K350 for ZF4-5 site")
        
        self.box_size = [20, 20, 20]
        print(f"  Box size: {self.box_size[0]} x {self.box_size[1]} x {self.box_size[2]} Å")
    
    def prepare_ligand(self, smiles, name, temp_dir):
        """Convert SMILES to PDBQT format"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
            return None, "3D embedding failed"
        
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        
        pdb_file = temp_dir / f"{name}.pdb"
        pdbqt_file = temp_dir / f"{name}.pdbqt"
        
        Chem.MolToPDBFile(mol, str(pdb_file))
        result = subprocess.run(f"obabel {pdb_file} -O {pdbqt_file}", 
                              shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not pdbqt_file.exists():
            return None, "PDBQT conversion failed"
        
        return str(pdbqt_file), None
    
    def dock_compound(self, pdbqt_file, site_name, center, output_pdbqt, exhaustiveness=32):
        """Dock a single compound to specified site"""
        v = Vina(sf_name='vina')
        v.set_receptor(self.receptor_pdbqt)
        v.compute_vina_maps(center=center, box_size=self.box_size)
        v.set_ligand_from_file(pdbqt_file)
        
        v.dock(exhaustiveness=exhaustiveness, n_poses=10)
        energies = v.energies(n_poses=10)
        
        # Extract best score
        score_raw = energies[0]
        if isinstance(score_raw, (list, tuple)):
            best_score = float(score_raw[0])
        elif isinstance(score_raw, np.ndarray):
            best_score = float(score_raw.flat[0])
        else:
            best_score = float(score_raw)
        
        v.write_poses(output_pdbqt, n_poses=10, overwrite=True)
        
        return best_score
    
    def process_compounds(self, compounds, exhaustiveness=32):
        """
        Process list of compounds for both sites
        
        Args:
            compounds: List of tuples (compound_id, smiles) or (compound_id, smiles, name)
            exhaustiveness: Vina exhaustiveness parameter (default=32, higher=more accurate but slower)
        
        Returns:
            results: List of docking results
        """
        results = []
        total = len(compounds)
        temp_dir = self.output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        print(f"\n[DOCKING] Processing {total} compounds x 2 sites = {total*2} docking runs")
        print(f"Exhaustiveness: {exhaustiveness}")
        print("="*70)
        
        for idx, compound_data in enumerate(compounds, 1):
            # Parse compound data
            if len(compound_data) == 2:
                comp_id, smiles = compound_data
                comp_name = comp_id
            else:
                comp_id, smiles, comp_name = compound_data
            
            print(f"\n[{idx}/{total}] {comp_name}")
            
            # Prepare ligand
            pdbqt_file, error = self.prepare_ligand(smiles, comp_name, temp_dir)
            if error:
                print(f"  ✗ Preparation failed: {error}")
                results.append({
                    'compound_id': comp_id,
                    'compound_name': comp_name,
                    'smiles': smiles,
                    'zf23_score': None,
                    'zf23_status': 'prep_failed',
                    'zf45_score': None,
                    'zf45_status': 'prep_failed',
                    'error': error
                })
                continue
            
            result = {
                'compound_id': comp_id,
                'compound_name': comp_name,
                'smiles': smiles
            }
            
            # Dock to ZF2-3
            try:
                output_23 = self.zf23_dir / f"{comp_name}_zf23.pdbqt"
                score_23 = self.dock_compound(pdbqt_file, 'ZF2-3', self.zf23_center, 
                                             str(output_23), exhaustiveness)
                result['zf23_score'] = score_23
                result['zf23_status'] = 'success'
                result['zf23_file'] = str(output_23)
                print(f"  ZF2-3: {score_23:.2f} kcal/mol")
            except Exception as e:
                result['zf23_score'] = None
                result['zf23_status'] = 'dock_failed'
                result['zf23_error'] = str(e)
                print(f"  ZF2-3: Failed - {e}")
            
            # Dock to ZF4-5
            try:
                output_45 = self.zf45_dir / f"{comp_name}_zf45.pdbqt"
                score_45 = self.dock_compound(pdbqt_file, 'ZF4-5', self.zf45_center, 
                                             str(output_45), exhaustiveness)
                result['zf45_score'] = score_45
                result['zf45_status'] = 'success'
                result['zf45_file'] = str(output_45)
                print(f"  ZF4-5: {score_45:.2f} kcal/mol")
            except Exception as e:
                result['zf45_score'] = None
                result['zf45_status'] = 'dock_failed'
                result['zf45_error'] = str(e)
                print(f"  ZF4-5: Failed - {e}")
            
            results.append(result)
        
        # Cleanup temp files
        for f in temp_dir.glob('*'):
            f.unlink()
        temp_dir.rmdir()
        
        return results
    
    def save_results(self, results, filename='docking_results.csv'):
        """Save results to CSV file"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'compound_id', 'compound_name', 'smiles',
                'zf23_score', 'zf23_status', 'zf23_file',
                'zf45_score', 'zf45_status', 'zf45_file',
                'best_site', 'best_score'
            ])
            writer.writeheader()
            
            for r in results:
                # Determine best site
                zf23 = r.get('zf23_score')
                zf45 = r.get('zf45_score')
                
                if zf23 is not None and zf45 is not None:
                    if zf23 < zf45:
                        r['best_site'] = 'ZF2-3'
                        r['best_score'] = zf23
                    else:
                        r['best_site'] = 'ZF4-5'
                        r['best_score'] = zf45
                elif zf23 is not None:
                    r['best_site'] = 'ZF2-3'
                    r['best_score'] = zf23
                elif zf45 is not None:
                    r['best_site'] = 'ZF4-5'
                    r['best_score'] = zf45
                else:
                    r['best_site'] = 'None'
                    r['best_score'] = None
                
                writer.writerow(r)
        
        print(f"\n✓ Results saved to: {output_file}")
        return output_file
    
    def print_summary(self, results):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        total = len(results)
        zf23_success = sum(1 for r in results if r.get('zf23_status') == 'success')
        zf45_success = sum(1 for r in results if r.get('zf45_status') == 'success')
        
        print(f"Total compounds: {total}")
        print(f"ZF2-3 successful: {zf23_success}/{total}")
        print(f"ZF4-5 successful: {zf45_success}/{total}")
        
        # Best scores
        zf23_scores = [r['zf23_score'] for r in results if r.get('zf23_score') is not None]
        zf45_scores = [r['zf45_score'] for r in results if r.get('zf45_score') is not None]
        
        if zf23_scores:
            print(f"\nZF2-3 Scores:")
            print(f"  Best: {min(zf23_scores):.2f} kcal/mol")
            print(f"  Mean: {np.mean(zf23_scores):.2f} kcal/mol")
            print(f"  Strong binders (<-7.0): {sum(1 for s in zf23_scores if s < -7.0)}")
        
        if zf45_scores:
            print(f"\nZF4-5 Scores:")
            print(f"  Best: {min(zf45_scores):.2f} kcal/mol")
            print(f"  Mean: {np.mean(zf45_scores):.2f} kcal/mol")
            print(f"  Strong binders (<-7.0): {sum(1 for s in zf45_scores if s < -7.0)}")
        
        print("="*70)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch dock compounds to GLI1 ZF2-3 and ZF4-5 sites')
    parser.add_argument('input_file', help='CSV file with columns: id,smiles (optional: name)')
    parser.add_argument('--exhaustiveness', type=int, default=32, 
                       help='Vina exhaustiveness (default=32, higher=slower but more accurate)')
    parser.add_argument('--output-dir', default='docking_results', 
                       help='Output directory (default=docking_results)')
    
    args = parser.parse_args()
    
    # Read compounds from CSV
    compounds = []
    with open(args.input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_id = row.get('id', row.get('compound_id', ''))
            smiles = row.get('smiles', row.get('SMILES', ''))
            name = row.get('name', comp_id)
            
            if smiles:
                compounds.append((comp_id, smiles, name))
    
    print(f"Loaded {len(compounds)} compounds from {args.input_file}")
    
    # Initialize docker
    docker = DualSiteBatchDocker(output_dir=args.output_dir)
    
    # Process compounds
    results = docker.process_compounds(compounds, exhaustiveness=args.exhaustiveness)
    
    # Save and summarize
    docker.save_results(results)
    docker.print_summary(results)


if __name__ == '__main__':
    main()
