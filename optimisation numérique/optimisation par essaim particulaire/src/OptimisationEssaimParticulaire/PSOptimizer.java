package OptimisationEssaimParticulaire;

import mapleConnection.Det_3_qutrits;
import mapleConnection.Det_Gr_4_8_fermions;
import mapleConnection.Discrim_4_qubits;
import mapleConnection.Discrim_5_qubits;
import mapleConnection.Discrim_6_qubits;
import mapleConnection.Discrim_7_qubits;
import mapleConnection.SphereFunction;

public class PSOptimizer {

	public static void main(String[] args) {
		
		SphereFunction test = new SphereFunction(); 
		Discrim_4_qubits discrim4qubits = new Discrim_4_qubits();
		Discrim_5_qubits discrim5qubits = new Discrim_5_qubits();
		Discrim_6_qubits discrim6qubits = new Discrim_6_qubits();	
		Discrim_7_qubits discrim7qubits = new Discrim_7_qubits();
		Det_3_qutrits det3qutrits = new Det_3_qutrits();
		Det_Gr_4_8_fermions detGr4_8_fermions = new Det_Gr_4_8_fermions();
		
		int nbParticules = 10000;
		int nbMovesMax = 5000;
		int numberCoordinates = 5;
		int voisinageSize = (int) (nbParticules/25.);
		
		double[][] variablesRange = new double[nbParticules][2];
		for (int i=0; i<nbParticules; i++) {
			variablesRange[i][0]=-1;
			variablesRange[i][1]=1;
		}
		Essaim essaim = new Essaim(nbParticules,voisinageSize,numberCoordinates,null,variablesRange,discrim4qubits,nbMovesMax);
		
		for (int i=0; i<nbMovesMax; i++) {
			essaim.evolution();
		}
		
	}

	
	
}
;