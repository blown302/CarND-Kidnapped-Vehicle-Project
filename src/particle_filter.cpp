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
#include <limits>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta) {

    // Create normal distributions
    normal_distribution<double> dist_x = getPositionNoiseDistributionX(x);
    normal_distribution<double> dist_y = getPositionNoiseDistributionY(y);
    normal_distribution<double> dist_theta = getPositionNoiseDistributionTheta(theta);

    for (int i = 0; i < num_particles_; ++i) {
        particles_.push_back(Particle {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1});
    }

    is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double velocity, double yaw_rate) {
    for (auto &particle : particles_) {
        if (abs(yaw_rate) < .001) {
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
    particle.y += (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    particle.theta += yaw_rate * delta_t;
}

vector<LandmarkObs> ParticleFilter::dataAssociation(Particle &particle, vector<LandmarkObs> &observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    vector<LandmarkObs> results(observations.size());
	for (auto &observation : observations) {
        auto transformedObservation = transformObservationToMapCoordinates(particle, observation);
        transformedObservation.id = findNearestNeighbor(transformedObservation);
        results.push_back(transformedObservation);
	}
	return results;
}

void ParticleFilter::updateWeights(double sensor_range, vector<LandmarkObs> &observations) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles_ are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (auto &particle : particles_) {
        auto associatedObservations = dataAssociation(particle, observations);
        particle.weight = calculateObservationWeights(observations);
    }


}

void ParticleFilter::resample() {
	// TODO: Resample particles_ with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    // TODO: pull out random to a func
    vector<Particle> particles;
	int index = static_cast<int>((rand() / (RAND_MAX)) * particles_.size());
    double beta = 0.;
    auto max = max_element(particles_.begin(), particles_.end(),
	        [](const Particle &a, const Particle &b){
               return a.weight < b.weight;
            })->weight;

	for (auto i = 0; i < particles_.size(); ++i) {
	    // TODO: pull out random to a func
	    beta += (rand() / (RAND_MAX)) * 2. * max;
	    while (beta > particles_[index].weight) {
            beta -= particles_[index].weight;
            index = static_cast<int>((index + 1) % particles_.size());
	    }
        particles.push_back(particles_[index]);
	}
    particles_ = particles;
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

void ParticleFilter::addMeasurementNoise(LandmarkObs observation) {
    observation.x = getLandmarkNoiseDistributionX(observation.x)(gen);
    observation.y = getLandmarkNoiseDistributionY(observation.y)(gen);
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

normal_distribution<double> ParticleFilter::getLandmarkNoiseDistributionX(double x) {
    return normal_distribution<double>(x, std_dev_land_x);
}

normal_distribution<double> ParticleFilter::getLandmarkNoiseDistributionY(double y) {
    return normal_distribution<double>(y, std_dev_land_y);
}

LandmarkObs ParticleFilter::transformObservationToMapCoordinates(Particle particle, LandmarkObs &observation) {
    double x = particle.x + (cos(particle.theta * observation.x)) - (sin(particle.theta) * observation.y);
    double y = particle.y + (sin(particle.theta * observation.x)) + (cos(particle.theta) * observation.y);
    return LandmarkObs{0, x, y};
}

int ParticleFilter::findNearestNeighbor(LandmarkObs observation) {
    double smallest_dist = numeric_limits<double>::infinity();
    int smallest_id = numeric_limits<int>::infinity();

    for (auto landmark : map_.landmark_list) {
        auto distance = dist(landmark.x_f, landmark.y_f, observation.x, observation.y);

        if (distance < smallest_dist) {
            smallest_dist = distance;
            smallest_id = landmark.id_i;
        }
    }

    return smallest_id;
}

double ParticleFilter::calculateObservationWeight(LandmarkObs &observation) {
    auto matchingLandmark = map_.landmark_list[observation.id];
    double gauss_norm = (1/(2 * M_PI * std_dev_land_x * std_dev_land_y));
    double exponent = (pow((observation.x - matchingLandmark.x_f), 2)/(2 * pow(std_dev_land_x, 2))
            + (pow(observation.y - matchingLandmark.y_f, 2)/2 * pow(std_dev_land_y, 2)));

    return gauss_norm * exp(exponent);
}

double ParticleFilter::calculateObservationWeights(vector<LandmarkObs> &observations) {
    double acc = 1;
    for (auto &observation : observations) {
        acc *= calculateObservationWeight(observation);
    }
    return acc;
}


