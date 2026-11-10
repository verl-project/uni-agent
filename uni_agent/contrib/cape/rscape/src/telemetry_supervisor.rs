// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Serialize, Deserialize)]
struct ProcessRecord { pid: i32, start_ticks: String, argv: Vec<String> }

fn process_record(pid: i32) -> Option<ProcessRecord> {
    let root = PathBuf::from(format!("/proc/{pid}"));
    if fs::metadata(&root).ok()?.uid() != unsafe {nix::libc::getuid()} {return None;}
    let stat = fs::read_to_string(root.join("stat")).ok()?;
    let fields: Vec<_> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    if fields.first() == Some(&"Z") {return None;}
    let bytes = fs::read(root.join("cmdline")).ok()?;
    let argv = String::from_utf8(bytes).ok()?.trim_end_matches('\0').split('\0').map(str::to_owned).collect();
    Some(ProcessRecord {pid, start_ticks: fields.get(19)?.to_string(), argv})
}

pub struct PrometheusSupervisor {
    root: PathBuf,
    scheduler_id: String,
    binary: String,
    pub query_url: Option<String>,
    pub receiver_url: Option<String>,
    child: Option<Child>,
    pub stop_on_drop: bool,
}

impl PrometheusSupervisor {
    pub fn new(state_dir: &Path, scheduler: &str, binary: String) -> Result<Self> {
        let root = fs::canonicalize(state_dir)?.parent().context("missing deployment parent")?.join("prometheus");
        Ok(Self {root, scheduler_id:scheduler.into(), binary, query_url:None, receiver_url:None, child:None, stop_on_drop:false})
    }

    fn lock(&self) -> Result<File> {
        match fs::DirBuilder::new().mode(0o700).create(&self.root) { Ok(()) => (), Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>(), Err(e)=>return Err(e.into()) }
        let meta = fs::symlink_metadata(&self.root)?;
        ensure!(meta.is_dir() && meta.uid()==unsafe{nix::libc::getuid()} && meta.mode() & 0o077 == 0, "Prometheus directory must be private and owned by current user");
        let lock = OpenOptions::new().read(true).write(true).create(true).mode(0o600).custom_flags(nix::libc::O_NOFOLLOW).open(self.root.join("owner.lock"))?;
        nix::fcntl::flock(lock.as_raw_fd(), nix::fcntl::FlockArg::LockExclusiveNonblock)?;
        Ok(lock)
    }

    fn write(&self, name: &str, value: &str) -> Result<()> {
        let temporary = self.root.join(format!(".publish-{}",uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).custom_flags(nix::libc::O_NOFOLLOW).open(&temporary)?;
            file.write_all(value.as_bytes())?; file.sync_all()?;
            fs::rename(&temporary,self.root.join(name))?; Ok(())
        })();
        let _ = fs::remove_file(temporary);
        result
    }

    fn owned(&self, record: &ProcessRecord) -> bool {
        record.argv.contains(&format!("--config.file={}",self.root.join("prometheus.yml").display()))
            && record.argv.contains(&format!("--storage.tsdb.path={}",self.root.join("data").display()))
    }

    fn discover(&self) -> Result<Option<ProcessRecord>> {
        let records: Vec<_> = fs::read_dir("/proc")?.filter_map(|p|p.ok()?.file_name().to_str()?.parse::<i32>().ok())
            .filter_map(process_record).filter(|r|self.owned(r)).collect();
        ensure!(records.len() <= 1,"multiple Prometheus processes claim this instance");
        Ok(records.into_iter().next())
    }

    fn signal(&self, record: &ProcessRecord, signal: i32) -> Result<()> {
        let fd = unsafe{nix::libc::syscall(nix::libc::SYS_pidfd_open, record.pid, 0)} as i32;
        if fd < 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error()==Some(nix::libc::ESRCH) {return Ok(());}
            if error.raw_os_error()==Some(nix::libc::ENOSYS) {
                if process_record(record.pid).map(|r|r.start_ticks==record.start_ticks && self.owned(&r)).unwrap_or(false) {
                    let result=unsafe{nix::libc::kill(record.pid,signal)};
                    if result != 0 {
                        let error=std::io::Error::last_os_error();
                        if error.raw_os_error()!=Some(nix::libc::ESRCH){return Err(error.into());}
                    }
                }
                return Ok(());
            }
            return Err(error.into());
        }
        let result = if process_record(record.pid).map(|r|r.start_ticks==record.start_ticks && self.owned(&r)).unwrap_or(false) {
            unsafe{nix::libc::syscall(nix::libc::SYS_pidfd_send_signal, fd, signal, std::ptr::null::<nix::libc::siginfo_t>(), 0)}
        } else {0};
        let error = std::io::Error::last_os_error();
        unsafe{nix::libc::close(fd)};
        if result < 0 && error.raw_os_error()!=Some(nix::libc::ESRCH) {return Err(error.into());}
        Ok(())
    }

    pub async fn start(&mut self) -> Result<Value> {
        let _lock = self.lock()?;
        let identity = self.root.join("instance.json");
        if identity.exists() {
            let value: Value = serde_json::from_slice(&fs::read(identity)?)?;
            ensure!(value["scheduler_id"]==self.scheduler_id,"Prometheus directory belongs to another scheduler");
        } else {self.write("instance.json",&json!({"scheduler_id":self.scheduler_id}).to_string())?;}
        let mut record = self.discover()?;
        if record.is_none() {
            self.write("prometheus.yml","global:\n  scrape_interval: 5s\nscrape_configs: []\n")?;
            let socket = std::net::TcpListener::bind("0.0.0.0:0")?;
            let port = socket.local_addr()?.port(); drop(socket);
            let mut command = Command::new(&self.binary);
            command.args([format!("--config.file={}",self.root.join("prometheus.yml").display()),
                format!("--storage.tsdb.path={}",self.root.join("data").display()),format!("--web.listen-address=0.0.0.0:{port}"),
                "--web.enable-remote-write-receiver".into(),"--storage.tsdb.retention.time=7d".into(),"--storage.tsdb.retention.size=1GB".into()])
                .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
            for (key,_) in std::env::vars_os() {if key.to_string_lossy().starts_with("CAPE_") {command.env_remove(key);}}
            unsafe {command.pre_exec(|| {if nix::libc::setsid()<0 {return Err(std::io::Error::last_os_error());} Ok(())});}
            self.child = Some(command.spawn()?);
            record = process_record(self.child.as_ref().unwrap().id() as i32);
        }
        let record = record.context("Prometheus did not start")?;
        let result = self.ready(&record).await;
        if result.is_err() {if let Some(child)=self.child.as_mut() {let _=child.kill();let _=child.wait();}}
        result
    }

    async fn ready(&mut self, record: &ProcessRecord) -> Result<Value> {
        let address = record.argv.iter().find_map(|a|a.strip_prefix("--web.listen-address=0.0.0.0:")).context("invalid owned Prometheus listener")?;
        let port: u16 = address.parse()?;
        let mut hostname = [0_u8;256];
        ensure!(unsafe{nix::libc::gethostname(hostname.as_mut_ptr().cast(),hostname.len())}==0,"cannot read hostname");
        let host = std::str::from_utf8(&hostname[..hostname.iter().position(|v|*v==0).unwrap_or(hostname.len())])?;
        self.query_url=Some(format!("http://127.0.0.1:{port}"));
        self.receiver_url=Some(format!("http://{host}:{port}/api/v1/write"));
        let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_millis(300)).build()?;
        let deadline=Instant::now()+Duration::from_secs(10);
        while Instant::now()<deadline {
            ensure!(process_record(record.pid).map(|r|r.start_ticks==record.start_ticks).unwrap_or(false),"Prometheus exited before ready");
            if client.get(format!("{}/-/ready",self.query_url.as_ref().unwrap())).send().await.map(|r|r.status().as_u16()==200).unwrap_or(false) {
                let mut manifest=serde_json::to_value(record)?;
                manifest["query_url"]=json!(self.query_url);manifest["receiver_url"]=json!(self.receiver_url);
                self.write("process.json",&manifest.to_string())?;
                return Ok(json!({"state":"ready","query_url":self.query_url,"receiver_url":self.receiver_url}));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        anyhow::bail!("Prometheus readiness timeout")
    }

    pub fn stop(&mut self) -> Result<()> {
        let _lock=self.lock()?;
        let identity: Value=serde_json::from_slice(&fs::read(self.root.join("instance.json"))?)?;
        ensure!(identity["scheduler_id"]==self.scheduler_id,"Prometheus ownership mismatch");
        if let Some(record)=self.discover()? {
            self.signal(&record,nix::libc::SIGTERM)?;
            let deadline=Instant::now()+Duration::from_secs(5);
            while process_record(record.pid).map(|r|r.start_ticks==record.start_ticks).unwrap_or(false) {
                if Instant::now()>=deadline {self.signal(&record,nix::libc::SIGKILL)?;break;}
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        if let Some(child)=self.child.as_mut(){let _=child.wait();}
        self.stop_on_drop=false;
        Ok(())
    }
}

impl Drop for PrometheusSupervisor {
    fn drop(&mut self){if self.stop_on_drop {if self.stop().is_err(){eprintln!("metrics: owned Prometheus cleanup failed");}}}
}
