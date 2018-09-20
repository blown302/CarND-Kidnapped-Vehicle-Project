#include <utility>

/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <random> // Need this for sampling from distributions

#include "helper_functions.h"

using namespace std;

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {
	
	// Number of particles_ to draw
	unsigned int num_particles;
	
	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles_
	std::vector<double> weights;
	
public:
	
	// Set of current particles_
	std::vector<Particle> particles_;

	// Constructor
	// @param num_particles Number of particles_
	ParticleFilter(const Map map, const double std_dev_pos[], const double std_dev_land[]) : num_particles(100), is_initialized(false) {
	    map_ = map;
	    std_dev_pos_x = std_dev_pos[0];
	    std_dev_pos_y = std_dev_pos[1];
	    std_dev_pos_theta = std_dev_pos[2];

	    std_dev_land_x = std_dev_land[0];
	    std_dev_land_y = std_dev_land[1];
	}

	// Destructor
	~ParticleFilter() = default;

    /**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double velocity, double yaw_rate);
	

	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, vector<LandmarkObs> &observations);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles_ list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);

	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}
private:
    Map map_;
	default_random_engine gen;
    double std_dev_pos_x;
    double std_dev_pos_y;
    double std_dev_pos_theta;
    double std_dev_land_x;
    double std_dev_land_y;

    /**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
    vector<LandmarkObs> dataAssociation(Particle& particle, vector<LandmarkObs>& observations);

    static void predictStraight(Particle &particle, double delta_t, double yaw_rate, double velocity);
    static void predictCurved(Particle &particle, double delta_t, double yaw_rate, double velocity);
    void addPositionNoise(Particle particle);
    void addMeasurementNoise(LandmarkObs observation);
    normal_distribution<double> getPositionNoiseDistributionX(double x);
    normal_distribution<double> getPositionNoiseDistributionY(double y);
    normal_distribution<double> getPositionNoiseDistributionTheta(double theta);
    normal_distribution<double> getLandmarkNoiseDistributionX(double x);
    normal_distribution<double> getLandmarkNoiseDistributionY(double y);
    LandmarkObs transformObservationToMapCoordinates(Particle particle, LandmarkObs &observation);
    int findNearestNeighbor(LandmarkObs observation);
    double calculateObservationWeight(LandmarkObs &observation);
    double calculateObservationWeights(vector<LandmarkObs> &observations);
};



#endif /* PARTICLE_FILTER_H_ */
