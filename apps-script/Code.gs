/**
 * Web app nhận JSON đánh giá từ review_site.html (GitHub Pages)
 * và lưu thành file trong một thư mục Google Drive.
 *
 * Deploy: Deploy → New deployment → Web app
 *   Execute as: Me
 *   Who has access: Anyone
 */
var FOLDER_ID = "1bZtOX1H0fdL_0XNuRqqvvIlBJLOay3PL";

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return jsonOut({ ok: false, error: "empty-body" });
    }

    var data = JSON.parse(e.postData.contents);
    var namePart = "reviewer";
    if (data.reviewer && data.reviewer.name) {
      namePart = String(data.reviewer.name)
        .replace(/[\\/:*?"<>|]+/g, "")
        .replace(/\s+/g, "_")
        .substring(0, 60) || "reviewer";
    }

    var stamp = Utilities.formatDate(new Date(), "Asia/Ho_Chi_Minh", "yyyyMMdd_HHmmss");
    var filename = "cgcn_review_" + namePart + "_" + stamp + ".json";
    var folder = DriveApp.getFolderById(FOLDER_ID);

    folder.createFile(
      filename,
      JSON.stringify(data, null, 2),
      MimeType.PLAIN_TEXT
    );

    return jsonOut({ ok: true, filename: filename });
  } catch (err) {
    return jsonOut({ ok: false, error: String(err) });
  }
}

function doGet() {
  return jsonOut({ ok: true, service: "cgcn-review-intake" });
}

function jsonOut(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
