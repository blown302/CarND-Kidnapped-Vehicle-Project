//
// Created by Thomas Milas on 9/20/18.
//

#include "test.h"

#define CATCH_CONFIG_MAIN

#include "lib/catch.hpp"
#include "particle_filter.h"
#include <limits>
#include <cmath>

using namespace std;

Map getLoadedMap() {
    vector<Map::single_landmark_s> landmarks {
        {1, 5, 3},
        {2, 2, 1},
        {3, 6, 1},
        {4, 7, 4},
        {5, 4, 7}
    };

    return {landmarks};
}


SCENARIO("initialize particle filter") {
    ParticleFilter particle_filter(Map{}, 100, (double[3]){0.3, 0.3, 0.01}, (double[2]){0.3, 0.3});
    WHEN("called with 100 items") {
        particle_filter.init(1,1,.1);

        THEN("update particles with 100 particles") {
            REQUIRE(particle_filter.particles_.size() == 100);
        }

        THEN("filter is initialized") {
            REQUIRE(particle_filter.initialized());
        }
    }
};

SCENARIO("predict particle at x=102 y=65 theta=(5*pi)/8") {
    ParticleFilter particle_filter(Map{}, 1, (double[3]){0.3, 0.3, 0.01}, (double[2]){0.3, 0.3});
    Particle particle{0, 102, 65,(5*M_PI)/8};
    particle_filter.particles_.push_back(particle);

    WHEN("velocity=110 yaw_rate=pi/8 delta_t=.1") {
        particle_filter.prediction(.1, 110, M_PI/8);

        THEN("sets particle x=97.59 y=75.08 yaw=(51*pi)/80") {
            auto p = particle_filter.particles_[0];

            double theta_approx((51*M_PI)/80);
            REQUIRE(abs(p.x - 97.59)/97.59 <= .3);
            REQUIRE(abs(p.x - 75.08)/75.08 <= .3);
            REQUIRE(abs(p.theta - theta_approx)/theta_approx < .1);
        }
    }
};

SCENARIO("transform observations into map coordinates") {
    ParticleFilter particle_filter(getLoadedMap(), 1, (double[3]){0.3, 0.3, 0.01}, (double[2]){0.3, 0.3});
    particle_filter.init(1, 1, .1);
    Particle particle{0, 4, 5, (-M_PI)/2};
    particle_filter.particles_[0] = particle;
    vector<LandmarkObs> observations{{0, 2, 2}, {0, 3, -2}, {0, 0, -4}};
    CHECK(particle_filter.particles_.size() == 1);
    CHECK(observations.size() == 3);

    WHEN("update is called with x=4, y=5 yaw=-M_PI/2") {
        particle_filter.updateWeights(numeric_limits<double>::infinity(), observations);

        auto p = particle_filter.particles_[0];
        THEN("observation 1 sets x=6 y=3") {
            REQUIRE(p.sense_x[0] == 6);
            REQUIRE(p.sense_y[0] == 3);
        }

        THEN("observation 2 sets x=2 y=2") {
            REQUIRE(p.sense_x[1] == 2);
            REQUIRE(round(p.sense_y[1]) == 2);
        }

        THEN("observation 3 sets x=0 y=5") {
            REQUIRE(p.sense_x[2] == 0);
            REQUIRE(p.sense_y[2] == 5);
        }
    }
};

SCENARIO("associate landmark to observation") {
    ParticleFilter particle_filter(getLoadedMap(), 1, (double[3]){0.3, 0.3, 0.01}, (double[2]){0.3, 0.3});
    particle_filter.init(1, 1, .1);
    Particle particle{0, 4, 5, (-M_PI)/2};
    particle_filter.particles_[0] = particle;
    vector<LandmarkObs> observations{{0, 2, 2}, {0, 3, -2}, {0, 0, -4}};
    CHECK(particle_filter.particles_.size() == 1);
    CHECK(observations.size() == 3);

    particle_filter.updateWeights(numeric_limits<double>::infinity(), observations);

    auto p = particle_filter.particles_[0];
    WHEN("update is called") {

        THEN("obs1 is associated with landmark 1") {
            REQUIRE(p.associations[0] == 1);
        }

        THEN("obs2 is associated with landmark 2") {
            REQUIRE(p.associations[1] == 2);
        }

        THEN("obs3 is associated with landmark 5") {
            REQUIRE(p.associations[2] == 2);
        }
    }
};

SCENARIO("update final weight") {
    ParticleFilter particle_filter(getLoadedMap(), 1, (double[3]){0.3, 0.3, 0.01}, (double[2]){0.3, 0.3});
    particle_filter.init(1, 1, .1);
    Particle particle{0, 4, 5, (-M_PI)/2};
    particle_filter.particles_[0] = particle;
    vector<LandmarkObs> observations{{0, 2, 2}, {0, 3, -2}, {0, 0, -4}};
    CHECK(particle_filter.particles_.size() == 1);
    CHECK(observations.size() == 3);

    particle_filter.updateWeights(numeric_limits<double>::infinity(), observations);

    auto p = particle_filter.particles_[0];
    WHEN("update is called ") {
        THEN("weight=4.60E-53") {
            double round_factor = pow(10, 55);
            REQUIRE(round(p.weight * round_factor)/round_factor == 4.60e-53);
        }
    }
}
