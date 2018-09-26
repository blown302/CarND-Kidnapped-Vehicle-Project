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
        weights_.push_back(1);
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
    particle.x += velocity * delta_t * cos(particle.theta);
    particle.y += velocity * delta_t * sin(particle.theta);
}

void ParticleFilter::predictCurved(Particle &particle, double delta_t, double yaw_rate, double velocity) {
    particle.x += (velocity/yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
    particle.y += (velocity/yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
    particle.theta += yaw_rate * delta_t;
}

void ParticleFilter::dataAssociation(Particle &particle, vector<LandmarkObs> &observations) {
    particle.sense_x = vector<double>(observations.size());
    particle.sense_y = vector<double>(observations.size());
    particle.associations = vector<int>(observations.size());
	for (auto i = 0; i < observations.size(); ++i) {
        auto transformedObservation = transformObservationToMapCoordinates(particle, observations[i]);

        particle.sense_x[i] = transformedObservation.x;
        particle.sense_y[i] = transformedObservation.y;
        auto nearest_id = findNearestNeighbor(transformedObservation);
        particle.associations[i] = nearest_id;
	}
}

void ParticleFilter::updateWeights(vector<LandmarkObs> &observations) {
	for (auto i = 0; i < num_particles_; ++i) {
        dataAssociation(particles_[i], observations);
        auto total_weight = calculateObservationWeights(particles_[i]);
        particles_[i].weight = total_weight;
        weights_[i] = total_weight;
	}
}

void ParticleFilter::resample() {
    discrete_distribution<int> weight_distribution(weights_.begin(), weights_.end());
    vector<Particle> sampled_particles(num_particles_);

    for (auto i = 0; i < num_particles_; ++i) {
        sampled_particles[i] = (particles_[weight_distribution(gen)]);
    }
    particles_ = sampled_particles;
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

void ParticleFilter::addPositionNoise(Particle &particle) {
    particle.x = getPositionNoiseDistributionX(particle.x)(gen);
    particle.y = getPositionNoiseDistributionY(particle.y)(gen);
    particle.theta = getPositionNoiseDistributionTheta(particle.theta)(gen);
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

LandmarkObs ParticleFilter::transformObservationToMapCoordinates(Particle particle, LandmarkObs &observation) {
    double x = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
    double y = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
    return LandmarkObs{0, x, y};
}

int ParticleFilter::findNearestNeighbor(LandmarkObs observation) {
    double smallest_dist = numeric_limits<double>::infinity();
    int smallest_id = numeric_limits<int>::infinity();

    for (auto landmark : map_.landmark_list) {
        auto distance = dist(observation.x, observation.y, landmark.x_f, landmark.y_f);

        if (distance < smallest_dist) {
            smallest_dist = distance;
            smallest_id = landmark.id_i;
        }
    }

    return smallest_id;
}

double ParticleFilter::calculateObservationWeight(double sense_x, double sense_y, int association_id) {
    auto matchingLandmark = map_.landmark_list[association_id - 1];
    double gauss_norm = 1/(2 * M_PI * std_dev_land_x * std_dev_land_y);
    double exponent = (pow(sense_x - matchingLandmark.x_f, 2))/(2 * pow(std_dev_land_x, 2))
            + (pow(sense_y - matchingLandmark.y_f, 2))/((2 * pow(std_dev_land_y, 2)));
    return gauss_norm * exp(-exponent);
}

double ParticleFilter::calculateObservationWeights(Particle &particle) {
    double acc = 1;
    for (auto i = 0; i < particle.sense_x.size(); ++i) {
        acc *= calculateObservationWeight(particle.sense_x[i], particle.sense_y[i], particle.associations[i]);
    }
    return acc;
}


