# Stage65A-PUB3 - краткое резюме

Классификация:
`stage65a-pub3-refs-published-github-release-workflows-permission-missing`.

Ручное подтверждение отключения GitHub Actions принято только для данного
этапа. Дополнительные проверки показали ноль workflows, ноль запусков и два
защитных условия, разрешающих публикацию в PyPI только в исходном репозитории
SpacemiT.

Ветки, `main` и аннотированный тег опубликованы в GitHub и GitLab с точным
совпадением SHA. Создание черновика GitHub Release завершилось HTTP 403:
текущему API-токену не хватает права Workflows write для тега, содержащего
изменение `.github/workflows`. GitHub Release и GitLab Release не созданы,
файлы релиза не опубликованы. PyPI остался без версии `2.1.2+riscy.1`.

Исходники Banana, custom executor и `/data/ncnn` не изменены.

Требуется новый ограниченный этап после выдачи GitHub-токену прав Contents
write и Workflows write. Stage65B не разрешен.
