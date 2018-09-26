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
#include <iostream>

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
	unsigned int num_particles_;
	
	// Flag, if filter is initialized
	bool is_initialized_;
	
public:
	
	// Set of current particles_
	std::vector<Particle> particles_;

	// Set of weights relating to particle.
	std::vector<double> weights_;

	// Constructor
	// @param num_particles Number of particles_
	ParticleFilter(const Map map, const unsigned int num_particles, const double std_dev_pos[],
				   const double std_dev_land[], const double sensor_range) : is_initialized_(false) {
	    map_ = map;
	    sensor_range_ = sensor_range;
	    num_particles_ = num_particles;
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
	void updateWeights(vector<LandmarkObs> &observations);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized_;
	}
private:
    Map map_;
	default_random_engine gen;
    double std_dev_pos_x;
    double std_dev_pos_y;
    double std_dev_pos_theta;
    double std_dev_land_x;
    double std_dev_land_y;
    double sensor_range_;

    /**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param particle Particle to associate observations to
	 * @param observations Vector of landmark observations
	 */
    void dataAssociation(Particle& particle, vector<LandmarkObs>& observations);

	/**
	 * Predict location of particle when going in a straight line.
	 * @param particle Particle to predict
	 * @param delta_t Time elapsed since last time step
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
    static void predictStraight(Particle &particle, double delta_t, double yaw_rate, double velocity);
    /**
     * Predict location of particle when going trajectory is curved.
     * @param particle Particle to predict
     * @param delta_t Time elapsed since last time step
     * @param yaw_rate Velocity of car from t to t+1 [m/s]
     * @param velocity Yaw rate of car from t to t+1 [rad/s]
     */
    static void predictCurved(Particle &particle, double delta_t, double yaw_rate, double velocity);
    /**
     * Add positional noise to a particle.
     * @param particle Particle to add noise to.
     */
    void addPositionNoise(Particle &particle);
    /**
     * Gets normal_distribution for x coordinate.
     * @param x X coordinate to derive distribution
     * @return normal_distribution with x as mean.
     */
    normal_distribution<double> getPositionNoiseDistributionX(double x);
	/**
     * Gets normal_distribution for y coordinate.
     * @param y Y coordinate to derive distribution
     * @return normal_distribution with y as mean
     */
    normal_distribution<double> getPositionNoiseDistributionY(double y);
	/**
     * Gets normal_distribution for theta.
     * @param theta theta coordinate to derive distribution
     * @return normal_distribution with theta as mean
     */
    normal_distribution<double> getPositionNoiseDistributionTheta(double theta);
    /**
     * Transforms observation to map coordinates with respect to particle position and orientation.
     * @param particle Particle used to relate observation to
     * @param observation Observation to map to map coordinates
     * @return LandmarkObs Observation mapped to map coordinates
     */
    LandmarkObs transformObservationToMapCoordinates(Particle particle, LandmarkObs &observation);
    /**
     * Find closest map landmark to observation.
     * @param filtered_landmarks Landmarks within range
     * @param observation Observation to to find closest landmark
     * @return int Landmark id of the closest landmark
     */
    int findNearestNeighbor(vector<Map::single_landmark_s> filtered_landmarks, LandmarkObs observation);
    /**
     * Calculate weight for of an observation based on associated landmark.
     * @param sense_x X observation coordinate mapped to global coordinates
     * @param sense_y Y observation coordinate mapped to global coordinates
     * @param association_id Id of the closest landmark to the observation in global space
     * @return double weight of a single observation
     */
    double calculateObservationWeight(double sense_x, double sense_y, int association_id);
    /**
     * Calculate total weight for of a particle.
     * @param particle Particle with associated observations
     * @return double Total particle weight
     */
    double calculateObservationWeights(Particle &particle);
    /**
     * Filters out map landmarks that are out of sensor range.
     * @param particle Particle to use as point of reference
     * @return vector<Map::single_landmark_s> valid landmarks for this particle
     */
	vector<Map::single_landmark_s> getLandmarksWithinRange(const Particle &particle) const;
};



#endif /* PARTICLE_FILTER_H_ */
