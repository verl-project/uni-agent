// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_QUEUE_CAPACITY: usize = 256;
const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_BATCH_INTERVAL: Duration = Duration::from_millis(5);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Event {
    pub timestamp: f64,
    pub event_type: String,
    pub entity_id: String,
    pub state: Option<String>,
    pub reason: Option<String>,
    pub details: Value,
}

pub struct DurableReceipt {
    receiver: Receiver<std::result::Result<(), String>>,
}

impl DurableReceipt {
    pub fn wait(self) -> Result<()> {
        self.receiver
            .recv()
            .context("journal writer stopped before durable acknowledgement")?
            .map_err(anyhow::Error::msg)
    }
}

enum Command {
    Append(Event),
    Durable(Event, mpsc::Sender<std::result::Result<(), String>>),
    Flush(mpsc::Sender<std::result::Result<(), String>>),
    Events(mpsc::Sender<std::result::Result<Vec<Event>, String>>),
    Close(mpsc::Sender<std::result::Result<(), String>>),
}

pub struct Journal {
    sender: SyncSender<Command>,
    writer: Option<JoinHandle<()>>,
    error: Arc<Mutex<Option<String>>>,
    commits: Arc<AtomicU64>,
    events: Arc<AtomicU64>,
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Journal")
            .field("commit_count", &self.commit_count())
            .finish_non_exhaustive()
    }
}

impl Journal {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, DEFAULT_QUEUE_CAPACITY, DEFAULT_BATCH_SIZE)
    }

    pub fn open_with_limits(
        path: impl AsRef<Path>,
        queue_capacity: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let (sender, receiver) = mpsc::sync_channel(queue_capacity.max(1));
        let (ready_sender, ready_receiver) = mpsc::channel();
        let error = Arc::new(Mutex::new(None));
        let commits = Arc::new(AtomicU64::new(0));
        let events = Arc::new(AtomicU64::new(0));
        let writer_error = Arc::clone(&error);
        let writer_commits = Arc::clone(&commits);
        let writer_events = Arc::clone(&events);
        let writer = thread::Builder::new()
            .name("rscape-journal-writer".into())
            .spawn(move || {
                writer_main(
                    path,
                    receiver,
                    ready_sender,
                    writer_error,
                    writer_commits,
                    writer_events,
                    batch_size.max(1),
                )
            })?;
        ready_receiver
            .recv()
            .context("journal writer stopped during initialization")?
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            sender,
            writer: Some(writer),
            error,
            commits,
            events,
        })
    }

    pub fn append(&self, event: &Event) -> Result<()> {
        self.check_error()?;
        self.sender
            .send(Command::Append(event.clone()))
            .map_err(|_| anyhow!("journal writer stopped"))?;
        self.check_error()
    }

    pub fn append_durable(&self, event: &Event) -> Result<DurableReceipt> {
        self.check_error()?;
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(Command::Durable(event.clone(), sender))
            .map_err(|_| anyhow!("journal writer stopped"))?;
        Ok(DurableReceipt { receiver })
    }

    pub fn flush(&self) -> Result<()> {
        self.barrier()?.wait()
    }

    pub fn barrier(&self) -> Result<DurableReceipt> {
        self.check_error()?;
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(Command::Flush(sender))
            .map_err(|_| anyhow!("journal writer stopped"))?;
        Ok(DurableReceipt { receiver })
    }

    pub fn events(&self) -> Result<Vec<Event>> {
        self.check_error()?;
        let (sender, receiver) = mpsc::channel();
        self.sender
            .send(Command::Events(sender))
            .map_err(|_| anyhow!("journal writer stopped"))?;
        receiver
            .recv()
            .context("journal writer stopped during replay read")?
            .map_err(anyhow::Error::msg)
    }

    pub fn commit_count(&self) -> u64 {
        self.commits.load(Ordering::Relaxed)
    }

    pub fn event_count(&self) -> u64 {
        self.events.load(Ordering::Relaxed)
    }

    pub fn close(&mut self) -> Result<()> {
        if self.writer.is_none() {
            return self.check_error();
        }
        let (sender, receiver) = mpsc::channel();
        let send_result = self.sender.send(Command::Close(sender));
        let result = if send_result.is_ok() {
            receiver
                .recv()
                .context("journal writer stopped during close")?
                .map_err(anyhow::Error::msg)
        } else {
            Err(anyhow!("journal writer stopped"))
        };
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        result.and_then(|_| self.check_error())
    }

    fn check_error(&self) -> Result<()> {
        let error = self.error.lock().unwrap().clone();
        match error {
            Some(error) => Err(anyhow!("journal writer failed: {error}")),
            None => Ok(()),
        }
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn writer_main(
    path: PathBuf,
    receiver: Receiver<Command>,
    ready: mpsc::Sender<std::result::Result<(), String>>,
    shared_error: Arc<Mutex<Option<String>>>,
    commits: Arc<AtomicU64>,
    events_written: Arc<AtomicU64>,
    batch_size: usize,
) {
    let mut connection = match open_connection(&path) {
        Ok(connection) => {
            let _ = ready.send(Ok(()));
            connection
        }
        Err(error) => {
            let message = format!("{error:#}");
            *shared_error.lock().unwrap() = Some(message.clone());
            let _ = ready.send(Err(message));
            return;
        }
    };
    let mut carry = None;
    loop {
        let command = match carry.take() {
            Some(command) => command,
            None => match receiver.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        match command {
            Command::Append(event) => {
                let mut events = vec![event];
                while events.len() < batch_size {
                    match receiver.recv_timeout(DEFAULT_BATCH_INTERVAL) {
                        Ok(Command::Append(event)) => events.push(event),
                        Ok(command) => {
                            carry = Some(command);
                            break;
                        }
                        Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => {
                            break
                        }
                    }
                }
                if let Err(error) =
                    insert_events(&mut connection, &events, &commits, &events_written)
                {
                    record_error(&shared_error, &error);
                }
            }
            Command::Durable(event, reply) => {
                let result = insert_events(&mut connection, &[event], &commits, &events_written)
                    .map_err(|error| format!("{error:#}"));
                if let Err(error) = &result {
                    record_error(&shared_error, &anyhow!(error.clone()));
                }
                let _ = reply.send(result);
            }
            Command::Flush(reply) => {
                let result = current_error(&shared_error);
                let _ = reply.send(result);
            }
            Command::Events(reply) => {
                let result = current_error(&shared_error)
                    .and_then(|_| read_events(&connection).map_err(|error| format!("{error:#}")));
                let _ = reply.send(result);
            }
            Command::Close(reply) => {
                let result = current_error(&shared_error);
                let _ = reply.send(result);
                break;
            }
        }
    }
}

fn open_connection(path: &Path) -> Result<Connection> {
    let connection =
        Connection::open(path).with_context(|| format!("open journal {}", path.display()))?;
    connection.pragma_update(None, "journal_mode", "WAL")?;
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            state TEXT,
            reason TEXT,
            details TEXT NOT NULL
        );",
    )?;
    Ok(connection)
}

fn insert_events(
    connection: &mut Connection,
    events: &[Event],
    commits: &AtomicU64,
    events_written: &AtomicU64,
) -> Result<()> {
    if events.is_empty() {
        return Ok(());
    }
    let transaction = connection.transaction()?;
    {
        let mut statement = transaction.prepare(
            "INSERT INTO events(timestamp,event_type,entity_id,state,reason,details)
             VALUES (?1,?2,?3,?4,?5,?6)",
        )?;
        for event in events {
            statement.execute(params![
                event.timestamp,
                event.event_type,
                event.entity_id,
                event.state,
                event.reason,
                serde_json::to_string(&event.details)?,
            ])?;
        }
    }
    transaction.commit()?;
    commits.fetch_add(1, Ordering::Relaxed);
    events_written.fetch_add(events.len() as u64, Ordering::Relaxed);
    Ok(())
}

fn read_events(connection: &Connection) -> Result<Vec<Event>> {
    let mut statement = connection.prepare(
        "SELECT timestamp,event_type,entity_id,state,reason,details FROM events ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        let details: String = row.get(5)?;
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
            details,
        ))
    })?;
    let mut events = Vec::new();
    for row in rows {
        let (timestamp, event_type, entity_id, state, reason, details) = row?;
        events.push(Event {
            timestamp,
            event_type,
            entity_id,
            state,
            reason,
            details: serde_json::from_str(&details)?,
        });
    }
    Ok(events)
}

fn record_error(shared_error: &Mutex<Option<String>>, error: &anyhow::Error) {
    let mut shared = shared_error.lock().unwrap();
    if shared.is_none() {
        *shared = Some(format!("{error:#}"));
    }
}

fn current_error(shared_error: &Mutex<Option<String>>) -> std::result::Result<(), String> {
    match shared_error.lock().unwrap().clone() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(index: usize, event_type: &str) -> Event {
        Event {
            timestamp: index as f64,
            event_type: event_type.into(),
            entity_id: format!("req-{index:06}"),
            state: Some("FAILED".into()),
            reason: Some("command_failed".into()),
            details: json!({"failure_kind":"workload"}),
        }
    }

    #[test]
    fn shares_python_event_schema_and_drains_in_order() {
        let directory = tempfile::tempdir().unwrap();
        let mut journal = Journal::open(directory.path().join("journal.sqlite")).unwrap();
        let receipt = journal
            .append_durable(&event(1, "request_submitted"))
            .unwrap();
        receipt.wait().unwrap();
        for index in 2..12 {
            journal.append(&event(index, "request_state")).unwrap();
        }
        journal.flush().unwrap();
        let events = journal.events().unwrap();
        assert_eq!(events.len(), 11);
        assert_eq!(events[0].event_type, "request_submitted");
        assert_eq!(events[10].timestamp, 11.0);
        assert_eq!(events[10].details["failure_kind"], "workload");
        assert!(journal.commit_count() < events.len() as u64);
        journal.close().unwrap();
    }

    #[test]
    fn bounded_queue_applies_backpressure_without_dropping() {
        let directory = tempfile::tempdir().unwrap();
        let mut journal =
            Journal::open_with_limits(directory.path().join("journal.sqlite"), 1, 64).unwrap();
        for index in 0..100 {
            journal.append(&event(index, "request_state")).unwrap();
        }
        journal.close().unwrap();
        let mut reopened = Journal::open(directory.path().join("journal.sqlite")).unwrap();
        assert_eq!(reopened.events().unwrap().len(), 100);
        reopened.close().unwrap();
    }
}
