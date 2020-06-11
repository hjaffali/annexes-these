package OptimisationEssaimParticulaire;

import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;

import mapleConnection.ObjectiveFunctionReel;

/**
 * 
 * @author hamza.jaffali
 *
 */
public class Particule {
	
	public static double[] scalarMultiplicationDoubleArray(float scalar, double[] tab) {
		double[] newTab = new double[tab.length];
		for (int i=0; i<tab.length; i++) {
			newTab[i] = scalar*tab[i];
		}
		return newTab;
	}
	
	public static double[] addDoubleArrays(double[] tab1, double[] tab2) {
		if(tab1.length!=tab2.length) {
			System.err.println("try to add tab with different sizes");
		}
		double[] newTab = new double[tab1.length];
		for (int i=0; i<tab1.length; i++) {
			newTab[i] = tab1[i] + tab2[i];
		}
		return newTab;
	}
	
	public static double[] substractDoubleArrays(double[] tab1, double[] tab2) {
		if(tab1.length!=tab2.length) {
			System.err.println("try to substract tab with different sizes");
		}
		double[] newTab = new double[tab1.length];
		for (int i=0; i<tab1.length; i++) {
			newTab[i] = tab1[i] - tab2[i];
		}
		return newTab;
	}
	
	public static double[] normalizeDoubleArrayReel(double[] tab) {
		double[] newTab = new double[tab.length];
		double squaredSum = 0;
		for (int i=0; i<tab.length; i++) {
			squaredSum = squaredSum + tab[i]*tab[i];
		}
		double norm = Math.sqrt(squaredSum);
		for (int i=0; i<tab.length; i++) {
			newTab[i] = tab[i]/norm;
		}
		return newTab;
	}
	
	private float intertieCoef;
	
	private int nbCoordinates;
	
	private double[] actualPosition;
	private double actualFunctionEvaluation;
	private double[] actualSpeed;

	private int index;
	
	private ArrayList<Particule> voisinage;
	private Essaim particuleEssaim;
	
	private double[] bestPosition;
	private double bestFunctionEvaluation;
	private Particule bestActualVoisinageParticule;
	
	Random rand = new Random();
	
	public Particule(int indeX, Essaim essaim, int numberCoordinates, double[] initPosition, double[] initSpeed) {
		this.index = 0;
		this.intertieCoef = Essaim.particule_intertie;
		this.nbCoordinates = numberCoordinates;
		this.particuleEssaim = essaim;
		
		this.actualPosition = initPosition.clone();
		this.actualSpeed = initSpeed.clone();
		
		this.actualPosition = Particule.normalizeDoubleArrayReel(this.actualPosition);
		this.actualFunctionEvaluation = this.particuleEssaim.objectiveFunction.apply(this.actualPosition);
		
		this.bestPosition = this.actualPosition;
		this.bestFunctionEvaluation = this.actualFunctionEvaluation;
	}
	
	public Particule(Particule copyParticule) {
		this.actualFunctionEvaluation = copyParticule.actualFunctionEvaluation;
		this.actualPosition = copyParticule.actualPosition.clone();
		this.actualSpeed = copyParticule.actualSpeed.clone();
		this.bestActualVoisinageParticule = copyParticule.bestActualVoisinageParticule;
		this.bestFunctionEvaluation = copyParticule.bestFunctionEvaluation;
		this.bestPosition = copyParticule.bestPosition.clone();
		this.index = copyParticule.index;
		this.intertieCoef = copyParticule.intertieCoef;
		this.nbCoordinates = copyParticule.nbCoordinates;
		this.particuleEssaim = copyParticule.particuleEssaim;
		this.rand = new Random();
		this.voisinage = new ArrayList<Particule>(copyParticule.voisinage);
	}
	
	public void setVoisinage(ArrayList<Particule> list) {
		this.voisinage = new ArrayList<Particule>(list);
		this.computeBestVoisinageParticule();
	}
	
	public void setBestFunctionEvaluation(double bestEval) {
		this.bestFunctionEvaluation = bestEval;
	}
	
	public void setIntertieCoef(float intertie_coef) {
		this.intertieCoef = intertie_coef;
	}
	

	
	public void computeBestVoisinageParticule() {
		double bestVoisinageEvaluation = Double.NEGATIVE_INFINITY;
		Particule bestParticule = this.voisinage.get(0);
		for (Particule particule : this.voisinage) {
			if (particule.bestFunctionEvaluation > bestVoisinageEvaluation) {
				bestVoisinageEvaluation = particule.bestFunctionEvaluation;
				bestParticule = particule;
			}
		}
		this.bestActualVoisinageParticule = bestParticule;
	}
	
	public void move() {
		// mise a jour de la vitesse et de la position
		float bestPositionCoef = rand.nextFloat();
		float bestVoisinagePositionCoef = rand.nextFloat();
		double[] bestPositionDirection = Particule.substractDoubleArrays(this.bestPosition, this.actualPosition);
		double[] bestVoisinagePositionDirection = Particule.substractDoubleArrays(this.bestActualVoisinageParticule.actualPosition, this.actualPosition);
		this.actualSpeed = Particule.addDoubleArrays(Particule.scalarMultiplicationDoubleArray(this.intertieCoef, this.actualSpeed),
				Particule.addDoubleArrays(Particule.scalarMultiplicationDoubleArray(bestPositionCoef, bestPositionDirection), Particule.scalarMultiplicationDoubleArray(bestVoisinagePositionCoef, bestVoisinagePositionDirection)));
		this.actualPosition = Particule.addDoubleArrays(this.actualPosition, this.actualSpeed);
		this.actualPosition = Particule.normalizeDoubleArrayReel(this.actualPosition);
		
		// mise a jour de l'evaluation de la fonction objectif
		this.actualFunctionEvaluation = this.particuleEssaim.objectiveFunction.apply(this.actualPosition).doubleValue();
		this.particuleEssaim.nbFunctionEvaluation++;
		
		// mise a jour eventuelle de la meilleure solution de la particule
		if (this.actualFunctionEvaluation > this.bestFunctionEvaluation) {
			this.bestPosition = this.actualPosition.clone();
			this.bestFunctionEvaluation = this.actualFunctionEvaluation;
			
			// mise a jour eventuelle de la meilleure solution pour toutes les particules
			if (this.bestFunctionEvaluation > this.particuleEssaim.bestParticle.bestFunctionEvaluation) {
				this.particuleEssaim.bestParticle = this;
			}
		}
	}
	
	public double[] getActualPosition() {
		return this.actualPosition.clone();
	}
	
	public double[] getBestPosition() {
		return this.bestPosition.clone();
	}
	
	public double getActualFunctionEvaluation() {
		return this.actualFunctionEvaluation;
	}
	
	public double getBestFunctionEvaluation() {
		return this.bestFunctionEvaluation;
	}
	
}
