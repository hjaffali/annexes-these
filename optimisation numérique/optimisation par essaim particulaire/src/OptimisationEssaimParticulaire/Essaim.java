package OptimisationEssaimParticulaire;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import mapleConnection.ObjectiveFunctionReel;

public class Essaim {

	public static double min_facteur_vitesse_initiale = 0.8,
				  		 max_facteur_vitesse_initiale = 2.0;
	public static float  particule_intertie = (float) 0.75;
	
	private int essaimSize;
	private int voisinageSize;
	private ArrayList<Particule> particuleList;
	private int nbCoordinates;

	private double[][] variablesRange;
	private double[][] allInit;
	
	public ObjectiveFunctionReel objectiveFunction;
	
	public Particule bestParticle;
	
	public long nbMoves;
	public long nbMovesMax;
	
	public long nbFunctionEvaluation;
	
	Random rand = new Random();
	
	public Essaim(int essaimsize, int voisinagesize, int numberOfCoordintes, double[][] allinit, double[][] variablesrange,
			ObjectiveFunctionReel objectFunct, long nbMouvementsMaximum) {
		this.essaimSize = essaimsize;
		this.voisinageSize = voisinagesize;
		this.nbCoordinates = numberOfCoordintes;
		this.allInit = allinit;
		this.variablesRange = variablesrange;
		this.objectiveFunction = objectFunct;
		this.bestParticle = new Particule(-1, this, 0, new double[numberOfCoordintes], new double[numberOfCoordintes]);
		this.bestParticle.setBestFunctionEvaluation(Double.NEGATIVE_INFINITY);
		
		this.nbMovesMax = nbMouvementsMaximum;
		this.nbMoves = 0;
		this.nbFunctionEvaluation = 0;
		
		this.initializeEssaim();
		
		this.callback();
		
	}
	
	public void initializeEssaim() {
		this.particuleList = new ArrayList<Particule>();
		double[] initPosition = new double[this.nbCoordinates];
		double[] initVector = new double[this.nbCoordinates];
		
		// initialisation des positions et des vitesses initiales des particules
		for (int index=0; index<this.essaimSize; index++) {
			if(this.allInit == null) {
				for (int i=0; i<this.nbCoordinates; i++) {
					initPosition[i] = rand.nextDouble()*(this.variablesRange[index][1]-this.variablesRange[index][0]) + this.variablesRange[index][0];
					initVector[i] = (rand.nextDouble()*(Essaim.max_facteur_vitesse_initiale-Essaim.min_facteur_vitesse_initiale) + Essaim.min_facteur_vitesse_initiale)*(rand.nextDouble()*(this.variablesRange[index][1]-this.variablesRange[index][0]) + this.variablesRange[index][0]);
				}
				this.particuleList.add(new Particule(index,this,this.nbCoordinates,initPosition,initVector));
			} else {
				for (int i=0; i<this.nbCoordinates; i++) {
					initVector[i] = (rand.nextDouble()*(Essaim.max_facteur_vitesse_initiale-Essaim.min_facteur_vitesse_initiale) + Essaim.min_facteur_vitesse_initiale)*(rand.nextDouble()*(this.variablesRange[index][1]-this.variablesRange[index][0]) + this.variablesRange[index][0]);
				}
				this.particuleList.add(new Particule(index,this,this.nbCoordinates,this.allInit[index],initVector));
			}
			
			if (this.particuleList.get(index).getBestFunctionEvaluation() > this.bestParticle.getBestFunctionEvaluation()) {
				this.bestParticle = this.particuleList.get(index);
			}
		}
		
		// génération du voisinage en forme ligne, tous les voisins sont à droite de la particule dans l'ordre croissant 
		for (int index=0; index<this.essaimSize; index++) {
			ArrayList<Particule> voisinage = new ArrayList<Particule>();
			for (int i=0; i<this.voisinageSize; i++) {
				voisinage.add(this.particuleList.get((index+(i+1)) % this.particuleList.size()));
			}
			this.particuleList.get(index).setVoisinage(voisinage);
		}
		
		
	}
	
	public void evolution() {
		// déplacement des particles
		for (Particule particule : this.particuleList) {
			particule.move();
		}
				
		//  determination du meilleur voisin pour chaque particule
		for (Particule particule : this.particuleList) {
			particule.computeBestVoisinageParticule();
		}
		
		this.nbMoves = this.nbMoves +1;
		this.callback();
	}
	
	public void callback() {
		System.out.println("Valeur fonction : "+this.bestParticle.getBestFunctionEvaluation()+"\nSolution actuelle : "+ Arrays.toString(this.bestParticle.getBestPosition())+", Nb itérations : "+this.nbMoves+" / "+this.nbMovesMax);
	}
	
}
