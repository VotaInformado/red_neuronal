# Django
from rest_framework.routers import SimpleRouter
from django.http import HttpResponse
from django.contrib import admin
from django.urls import include, path
from django.shortcuts import redirect
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

# Views
from red_neuronal.views import PredictionViewSet

# Neural network data endpoints

router = SimpleRouter()

router.register(r"", PredictionViewSet, basename="votes")


def redirect_to_health(request):
    return redirect("/health/")


def health_check(request):
    return HttpResponse("OK")


schema_view = get_schema_view(
    openapi.Info(
        title="Your API",
        default_version="v1",
        description="Your API description",
        terms_of_service="https://www.yourapp.com/terms/",
        contact=openapi.Contact(email="contact@yourapp.com"),
        license=openapi.License(name="Your License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path("", redirect_to_health),
    path("admin/", admin.site.urls),
    path("health/", health_check, name="health_check"),
    path("api/", include(router.urls)),
    path(
        "swagger/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    path("swagger.json", schema_view.without_ui(cache_timeout=0), name="schema-json"),  # To download the swagger.json
]
