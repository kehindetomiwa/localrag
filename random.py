import logging

# 1. Configure the logging system
logging.basicConfig(
    level=logging.DEBUG, # Shows DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s (Line %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a custom logger for this module
logger = logging.getLogger("MyApp")

print("=== USING PRINT ===")
print("App starting up")
print("Connecting to database...")
print("Warning: Connection took longer than 2 seconds")
print("Failed to connect to database!\n")


print("=== USING LOGGER ===")
logger.debug("App starting up (useful for troubleshooting during dev)")
logger.info("Connecting to database...")
logger.warning("Connection took longer than 2 seconds")
logger.error("Failed to connect to database!")