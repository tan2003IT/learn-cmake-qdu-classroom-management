#include"self_learn.hpp"
#include"spdlog/spdlog.h"
#include"camera.hpp"
#include"sets.hpp"
#include"computer.hpp"
self_learn::self_learn(){
    SPDLOG_INFO("learn by you self");
}
self_learn::~self_learn(){
    SPDLOG_CRITICAL("leave");
}
void self_learn::init(){
    Camera camera("haikang");
    Sets sets(12);
    camera.video();
    sets.heating();
}