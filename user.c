/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020-2021 Zoltán Vörös
 *
 * Copyright (c) 2026 Bo Jakobsen, Roskilde University, Denmark
 *   - MD simulation functions 
 *   - Part of "MD-in-a-Box"
 *   - Based on the MD simulation concept by Ulf R. Pedersen (urp.dk/md)
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "user.h"

#if ULAB_HAS_USER_MODULE

//| """This module should hold arbitrary user-defined functions."""
//|


// Global state for random numbers
// seed is hard coded for now
static uint32_t rng_state = 12345;

// Fast xorshift32 random generator Fixes problems with rand() on pimoroni build
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

// Uniform random distribution [0, 1)
static mp_float_t rand_uniform(void) {
    return (mp_float_t)xorshift32() / (mp_float_t)4294967296.0;
}


// Internal force calc function - not exposed to Python
static void calc_forces_internal(
    mp_float_t *px, mp_float_t *py,
    mp_float_t *vx, mp_float_t *vy,
    mp_float_t *fx, mp_float_t *fy,
    mp_float_t cutoff_sq, mp_float_t c12, mp_float_t c6,
    mp_float_t box_x, mp_float_t box_y, mp_float_t eps_w,
    mp_float_t std, mp_float_t friction,
    mp_float_t fgx, mp_float_t fgy,
    size_t n_atoms
) {

  // Zero forces before accumulating
  for(size_t i = 0; i < n_atoms; i++){
      fx[i] = 0.0;
      fy[i] = 0.0;
  }


  /* Soft wall forces (1/x^6) */
  
  // Left wall (x -> 0)
  for(size_t i = 0; i < n_atoms; i++){
    mp_float_t dx = px[i] ;
    // Simple debug print
    //mp_printf(&mp_plat_print, "dx = %f\n", (double)dx);
    if (dx < (mp_float_t)0.5) {dx = (mp_float_t)0.5;}  // prevent singularity
    mp_float_t dx2 = dx * dx;  // x ^ 2
    mp_float_t dx6 = dx2 * dx2 * dx2;  // x ^ 6
    fx[i] += eps_w / dx6;
  }

  // Right wall (x -> box_x )
  for(size_t i = 0; i < n_atoms; i++){
    mp_float_t dx = (mp_float_t)box_x - px[i]  ;
    if (dx < 0.5) dx = 0.5;  // prevent singularity
    mp_float_t dx2 = dx * dx;  // x ^ 2
    mp_float_t dx6 = dx2 * dx2 * dx2;  // x ^ 6
    fx[i] -= eps_w / dx6;
  }

  // Top wall (y -> 0)
  for(size_t i = 0; i < n_atoms; i++){
    mp_float_t dy = py[i] ;
    if (dy < 0.5) dy = 0.5;  // prevent singularity
    mp_float_t dy2 = dy * dy;  // y ^ 2
    mp_float_t dy6 = dy2 * dy2 * dy2;  // y ^ 6
    fy[i] += eps_w / dy6;
  }


  // Bottom wall (y -> box_y)
  for(size_t i = 0; i < n_atoms; i++){
    mp_float_t dy = (mp_float_t)box_y - py[i];
    if (dy < 0.5) dy = 0.5;  // prevent singularity
    mp_float_t dy2 = dy * dy;  // y ^ 2
    mp_float_t dy6 = dy2 * dy2 * dy2;  // y ^ 6
    fy[i] -= eps_w / dy6;
  }

  // Calculate LJ forces for all pairs
    for(size_t i = 0; i < (size_t)n_atoms; i++){
      for(size_t j = i+1; j < (size_t)n_atoms; j++){

	mp_float_t dx = px[j] - px[i];
	mp_float_t dy = py[j] - py[i];
	mp_float_t r2 = dx * dx + dy* dy;

	if ((r2 < cutoff_sq) && (r2 > 0.01)){
	  mp_float_t r6 = r2 * r2 * r2;
	  mp_float_t r12 = r6 * r6;
	  mp_float_t pre = (c12 / r12 + c6 / r6) / r2;

        
	  mp_float_t f_x = pre * dx;
	  mp_float_t f_y = pre * dy;

	  fx[i] += f_x;
	  fy[i] += f_y;
	  fx[j] -= f_x;
	  fy[j] -= f_y;
        }
      }
    }
  //Apply Langevin thermostat: friction + random forces.
    if (friction > 0){ 
      for(size_t i = 0; i < n_atoms; i++){

        // Per component, σ=1 approximate Gaussian:
        mp_float_t noise_x = (mp_float_t)2.0 * (rand_uniform() + rand_uniform() + rand_uniform() -(mp_float_t)1.5);
        mp_float_t noise_y = (mp_float_t)2.0 * (rand_uniform() + rand_uniform() + rand_uniform() - (mp_float_t)1.5);

        // noise + friction is added
        fx[i] += std *noise_x - friction * vx[i];
        fy[i] += std *noise_y - friction * vy[i];
      }
    }
  
  // Apply gravity
  for(size_t i = 0; i < n_atoms; i++){
    fx[i] += fgx;
    fy[i] += fgy;
  }


  // Clamp forces to prevent overflow
  mp_float_t max_force = 1.0e6;  // tune this - big but not astronomical
  for(size_t i = 0; i < n_atoms; i++){
    if (fx[i] > max_force) fx[i] = max_force;
    if (fx[i] < -max_force) fx[i] = -max_force;
    if (fy[i] > max_force) fy[i] = max_force;
    if (fy[i] < -max_force) fy[i] = -max_force;
  }
  
}


// Leap frog step
static mp_obj_t user_lf_step(size_t n_args, const mp_obj_t *args) {
    // NB! No type tests, will crash hard!

  
/*  Call seq from Python 
    self.pos_x,            # array, float
    self.pos_y,            # array, float
    self.vel_x,            # array, float
    self.vel_y,            # array, float
    self.force_x,          # array, float (changed in place)
    self.force_y,          # array, float (changed in place)
    self.cutoff_sq,        # float  
    self.lj_c12,           # float 
    self.lj_c6,            # float
    self.box_size_x,       # int
    self.box_size_y,       # int
    self.epsilon_wall      # float
    self.box_vel           # array float
    self.dt,               # float
    self.std,              # float
    self.langevin_friction,# float
    self.gravity_x,        # float
    self.gravity_y,        # float
    self.stride            # int
   */
  
  
  // handle input data
  ndarray_obj_t *px_obj = MP_OBJ_TO_PTR(args[0]);
  ndarray_obj_t *py_obj = MP_OBJ_TO_PTR(args[1]);
  ndarray_obj_t *vx_obj = MP_OBJ_TO_PTR(args[2]);
  ndarray_obj_t *vy_obj = MP_OBJ_TO_PTR(args[3]);
  ndarray_obj_t *fx_obj = MP_OBJ_TO_PTR(args[4]);
  ndarray_obj_t *fy_obj = MP_OBJ_TO_PTR(args[5]);
  mp_float_t cutoff_sq = mp_obj_get_float(args[6]);
  mp_float_t c12 = mp_obj_get_float(args[7]); 
  mp_float_t c6 = mp_obj_get_float(args[8]); 
  mp_int_t box_x = mp_obj_get_int(args[9]);
  mp_int_t box_y = mp_obj_get_int(args[10]);
  mp_float_t eps_w = mp_obj_get_float(args[11]);
  ndarray_obj_t *box_vel_obj = MP_OBJ_TO_PTR(args[12]);
  mp_float_t dt = mp_obj_get_float(args[13]);
  mp_float_t std = mp_obj_get_float(args[14]);
  mp_float_t friction = mp_obj_get_float(args[15]);
  mp_float_t fgx = mp_obj_get_float(args[16]);
  mp_float_t fgy = mp_obj_get_float(args[17]);
  mp_int_t stride = mp_obj_get_int(args[18]);
  

  /* // get pointers to array data */
  mp_float_t *px = (mp_float_t *)px_obj->array;
  mp_float_t *py = (mp_float_t *)py_obj->array;
  mp_float_t *vx = (mp_float_t *)vx_obj->array;
  mp_float_t *vy = (mp_float_t *)vy_obj->array;
  mp_float_t *fx = (mp_float_t *)fx_obj->array;
  mp_float_t *fy = (mp_float_t *)fy_obj->array;
  mp_float_t *box_vel = (mp_float_t *)box_vel_obj->array;
  
  // number of atoms
  size_t n_atoms = px_obj->len;

  // perform stride LF steps before returning
  for (size_t kk = 0; kk < (size_t)stride; kk++){
  
    // Step 1: Update positions
    for(size_t i = 0; i < n_atoms; i++){
      px[i] += vx[i] * dt - box_vel[0];
      py[i] += vy[i] * dt - box_vel[1];
    }

    // Emergency hard wall, hinder particles from escaping
    mp_float_t margin = 1.0;
    for(size_t i = 0; i < n_atoms; i++){
      if (px[i] < margin) { px[i] = margin; vx[i] = fabs(vx[i]); }
      if (px[i] > box_x - margin) { px[i] = box_x - margin; vx[i] = -fabs(vx[i]); }
      if (py[i] < margin) { py[i] = margin; vy[i] = fabs(vy[i]); }
      if (py[i] > box_y - margin) { py[i] = box_y - margin; vy[i] = -fabs(vy[i]); }
    }

    // step 2: update the forces from LJ interaction and walls (in place)
    calc_forces_internal(px, py, vx, vy, fx, fy, cutoff_sq, c12, c6, box_x, box_y, eps_w, std, friction, fgx, fgy, n_atoms);
																	      
    // Step 3: Update velocities
    for(size_t i = 0; i < n_atoms; i++){
      vx[i] += fx[i] * dt;
      vy[i] += fy[i] * dt;
    }
  }
    // at the end, return noting as we do inline change to forces
    return mp_const_none;
}

MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(user_lf_step_obj,0 ,20, user_lf_step);


// common to all functions

// All exposed functions must be in this structure
static const mp_rom_map_elem_t ulab_user_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_user) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_lf_step), (mp_obj_t)&user_lf_step_obj },
};

static MP_DEFINE_CONST_DICT(mp_module_ulab_user_globals, ulab_user_globals_table);

const mp_obj_module_t ulab_user_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_user_globals,
};
#if CIRCUITPY_ULAB
MP_REGISTER_MODULE(MP_QSTR_ulab_dot_user, ulab_user_module);
#endif
#endif

