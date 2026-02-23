use tauri::{
  api::version,
  command,
  utils::assets::EmbeddedAssets,
  Context,
  AboutMetadata,
  Manager,  
  PhysicalSize,
  Size,
  WindowBuilder,
};
use std::process::Command;

#[command]
fn open_signal_window(app: tauri::AppHandle) {
  let main_window = app.get_window("main").unwrap();
  let _ = main_window.set_size(Size::Physical(PhysicalSize { width: 1200, height: 800 }));
  
  let _signal_window = WindowBuilder::new(&app, "signals", tauri::WindowUrl::App("/#/signals".into()))
    .title("Trading Signals")
    .inner_size(800.0, 600.0)
    .build()
    .unwrap();
    
  let _chart_window = WindowBuilder::new(&app, "charts", tauri::WindowUrl::App("/#/charts".into()))
    .title("Trading Charts")
    .inner_size(1000.0, 700.0)
    .build()
    .unwrap();
    
  let _positions_window = WindowBuilder::new(&app, "positions", tauri::WindowUrl::App("/#/positions".into()))
    .title("Positions & P&L")
    .inner_size(600.0, 400.0)
    .build()
    .unwrap();
}

fn main() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![open_signal_window])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
