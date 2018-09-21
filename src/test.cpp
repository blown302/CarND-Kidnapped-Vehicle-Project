//
// Created by Thomas Milas on 9/20/18.
//

#include "test.h"

#define CATCH_CONFIG_MAIN

#include "lib/catch.hpp"
#include "particle_filter.h"
#include <cmath>




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
    Particle particle{0, 102, 65, (5*M_PI)/8};
    particle_filter.particles_.push_back(particle);

    WHEN("velocity=110 yaw_rate=pi/8 delta_t=.1") {
        particle_filter.prediction(.1, 110, M_PI/8);

        THEN("sets particle x=97.59 y=75.08 yaw=(51*pi)/80") {
            auto p = particle_filter.particles_[0];
            REQUIRE(round(p.x *100)/100 == 97.59);
            REQUIRE(round(p.y*100)/100 == 75.08);
            REQUIRE(p.theta == (51*M_PI)/80);
        }
    }
};
