// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agent;
mod auth;
mod identity;
mod management;
mod management_sso;
mod identity_nodes;
mod node_enrollment;
mod instance;
mod credential_file;
mod session_identity;
mod cli;
mod controller;
mod http;
mod journal;
mod model;
mod runtime;
mod workspace;
mod workspace_fs;
mod workspace_agent;
mod workspace_reclaim;
mod workspace_admission;
mod telemetry_contract;
mod telemetry_control;
mod telemetry_query;
mod telemetry_supervisor;
mod telemetry_sender;
mod telemetry_service;
mod telemetry_link;

#[tokio::main]
async fn main() {
    let code = match cli::run().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error:#}");
            1
        }
    };
    std::process::exit(code);
}
