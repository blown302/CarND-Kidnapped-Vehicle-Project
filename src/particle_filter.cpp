/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    std_dev_pos_x = std[0];
    std_dev_pos_y = std[1];
    std_dev_pos_theta = std[2];

    // Create normal distributions
    normal_distribution<double> dist_x = getPositionNoiseDistributionX(x);
    normal_distribution<double> dist_y = getPositionNoiseDistributionY(y);
    normal_distribution<double> dist_theta = getPositionNoiseDistributionTheta(theta);

    for (int i = 0; i < num_particles; ++i) {
        particles.push_back(Particle {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1});
    }

    weights = vector<double>(num_particles, 1);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double velocity, double yaw_rate) {
    for (Particle &particle : particles) {
        if (abs(particle.theta) < .001) {
            predictStraight(particle, delta_t, yaw_rate, velocity);
        } else {
            predictCurved(particle, delta_t, yaw_rate, velocity);
        }
        addPositionNoise(particle);
    }
}

void ParticleFilter::predictStraight(Particle &particle, double delta_t, double yaw_rate, double velocity) {
    particle.x += velocity * delta_t * cos(yaw_rate);
    particle.y += velocity * delta_t * sin(yaw_rate);
}

void ParticleFilter::predictCurved(Particle &particle, double delta_t, double yaw_rate, double velocity) {
    particle.x += (velocity/yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
    particle.y += (velocity/yaw_rate) * (cosh(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    particle.theta += yaw_rate * delta_t;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::addPositionNoise(Particle particle) {
    particle.x += getPositionNoiseDistributionX(particle.x)(gen);
    particle.y += getPositionNoiseDistributionY(particle.y)(gen);
    particle.theta += getPositionNoiseDistributionTheta(particle.theta)(gen);
}

normal_distribution<double> ParticleFilter::getPositionNoiseDistributionX(double x) {
    return normal_distribution<double>(x, std_dev_pos_x);
}

normal_distribution<double> ParticleFilter::getPositionNoiseDistributionY(double y) {
    return normal_distribution<double>(y, std_dev_pos_y);
}

normal_distribution<double> ParticleFilter::getPositionNoiseDistributionTheta(double theta) {
    return normal_distribution<double>(theta, std_dev_pos_theta);
}
